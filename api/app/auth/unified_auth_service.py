"""
Unified Authentication Service
Enterprise-grade centralized authentication with JWT, RBAC, SSO, and MFA
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import hashlib
import secrets
import base64
import hmac
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet
import jwt
from jwt.algorithms import RSAAlgorithm
import bcrypt
import pyotp
import qrcode
from io import BytesIO
import redis.asyncio as redis
import asyncpg
from sqlalchemy import create_engine, Column, String, DateTime, Text, Boolean, Integer, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY
import pydantic
from pydantic import BaseModel, Field, EmailStr
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import httpx
from saml2 import BINDING_HTTP_POST, BINDING_HTTP_REDIRECT
from saml2.client import Saml2Client
from saml2.config import Config as Saml2Config
import ldap3
from ldap3 import Server, Connection, ALL, SUBTREE
import xml.etree.ElementTree as ET
from authlib.integrations.starlette_client import OAuth
from authlib.oauth2 import OAuth2Error
import consul

logger = logging.getLogger(__name__)

class AuthMethod(Enum):
    PASSWORD = "password"
    MFA_TOTP = "mfa_totp"
    MFA_SMS = "mfa_sms"
    SAML_SSO = "saml_sso"
    OAUTH2 = "oauth2"
    LDAP = "ldap"
    API_KEY = "api_key"
    SERVICE_MESH = "service_mesh"

class Permission(Enum):
    # Chat permissions
    CHAT_READ = "chat:read"
    CHAT_WRITE = "chat:write"
    CHAT_DELETE = "chat:delete"
    CHAT_ADMIN = "chat:admin"
    
    # User permissions
    USER_READ = "user:read"
    USER_WRITE = "user:write"
    USER_DELETE = "user:delete"
    USER_ADMIN = "user:admin"
    
    # Analytics permissions
    ANALYTICS_READ = "analytics:read"
    ANALYTICS_WRITE = "analytics:write"
    ANALYTICS_ADMIN = "analytics:admin"
    
    # System permissions
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_CONFIG = "system:config"
    SYSTEM_MONITOR = "system:monitor"
    
    # Billing permissions
    BILLING_READ = "billing:read"
    BILLING_WRITE = "billing:write"
    BILLING_ADMIN = "billing:admin"

class Role(Enum):
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    MANAGER = "manager"
    USER = "user"
    VIEWER = "viewer"
    SERVICE = "service"

# Pydantic models
class UserLogin(BaseModel):
    email: EmailStr
    password: str
    mfa_code: Optional[str] = None
    remember_me: bool = False

class UserRegistration(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    first_name: str
    last_name: str
    organization: Optional[str] = None

class MFASetup(BaseModel):
    method: str = Field(..., regex="^(totp|sms)$")
    phone_number: Optional[str] = None

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    scope: str

class RoleAssignment(BaseModel):
    user_id: str
    role: Role
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    expires_at: Optional[datetime] = None

# SQLAlchemy models
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    organization = Column(String)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    mfa_enabled = Column(Boolean, default=False)
    mfa_secret = Column(String)
    phone_number = Column(String)
    last_login = Column(DateTime)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    role_assignments = relationship("UserRole", back_populates="user")
    sessions = relationship("UserSession", back_populates="user")

class UserRole(Base):
    __tablename__ = "user_roles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    role = Column(String, nullable=False)
    resource_type = Column(String)
    resource_id = Column(String)
    granted_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="role_assignments", foreign_keys=[user_id])
    granter = relationship("User", foreign_keys=[granted_by])

class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    session_token = Column(String, unique=True, nullable=False, index=True)
    refresh_token = Column(String, unique=True, nullable=False, index=True)
    device_info = Column(JSON)
    ip_address = Column(String)
    user_agent = Column(Text)
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="sessions")

class ServiceAccount(Base):
    __tablename__ = "service_accounts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    description = Column(Text)
    api_key = Column(String, unique=True, nullable=False, index=True)
    permissions = Column(ARRAY(String))
    rate_limit = Column(Integer, default=1000)
    is_active = Column(Boolean, default=True)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime)

class AuditLog(Base):
    __tablename__ = "auth_audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    action = Column(String, nullable=False)
    resource = Column(String)
    ip_address = Column(String)
    user_agent = Column(Text)
    success = Column(Boolean, nullable=False)
    details = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

class JWTKeyPair(Base):
    __tablename__ = "jwt_keypairs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key_id = Column(String, unique=True, nullable=False, index=True)
    public_key = Column(Text, nullable=False)
    private_key = Column(Text, nullable=False)
    algorithm = Column(String, default="RS256")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)

@dataclass
class AuthContext:
    user_id: str
    email: str
    roles: List[str]
    permissions: Set[Permission]
    session_id: str
    auth_method: AuthMethod
    mfa_verified: bool
    expires_at: datetime

class UnifiedAuthService:
    """
    Enterprise-grade unified authentication service with comprehensive security features
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Database
        self.engine = create_engine(config['database_url'])
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Redis for session management
        self.redis = redis.from_url(config['redis_url'])
        
        # JWT configuration
        self.jwt_issuer = config.get('jwt_issuer', 'ai-chatbot-auth')
        self.jwt_audience = config.get('jwt_audience', 'ai-chatbot-api')
        self.access_token_expire = config.get('access_token_expire', 3600)  # 1 hour
        self.refresh_token_expire = config.get('refresh_token_expire', 2592000)  # 30 days
        
        # Key rotation
        self.current_keypair: Optional[Dict[str, Any]] = None
        self.keypairs: Dict[str, Dict[str, Any]] = {}
        
        # Role-based access control
        self.role_permissions = self._initialize_rbac()
        
        # OAuth2 providers
        self.oauth = OAuth()
        self._setup_oauth_providers()
        
        # SAML configuration
        self.saml_clients: Dict[str, Saml2Client] = {}
        self._setup_saml_providers()
        
        # LDAP configuration
        self.ldap_servers: Dict[str, Dict[str, Any]] = {}
        self._setup_ldap_servers()
        
        # Service mesh authentication
        self.consul_client = consul.Consul(
            host=config.get('consul_host', 'localhost'),
            port=config.get('consul_port', 8500)
        )
        
        # Rate limiting
        self.rate_limits = {
            'login': {'requests': 5, 'window': 300},  # 5 attempts per 5 minutes
            'mfa': {'requests': 3, 'window': 300},    # 3 attempts per 5 minutes
            'api': {'requests': 1000, 'window': 3600} # 1000 requests per hour
        }
        
    async def initialize(self):
        """Initialize the authentication service"""
        
        # Load or generate JWT keypairs
        await self._initialize_jwt_keys()
        
        # Start background tasks
        asyncio.create_task(self._key_rotation_task())
        asyncio.create_task(self._session_cleanup_task())
        asyncio.create_task(self._audit_cleanup_task())
        
        logger.info("Unified Authentication Service initialized")
        
    async def _initialize_jwt_keys(self):
        """Initialize JWT signing keys with rotation"""
        try:
            db = self.SessionLocal()
            try:
                # Load active keypairs
                active_keys = db.query(JWTKeyPair).filter(
                    JWTKeyPair.is_active == True,
                    JWTKeyPair.expires_at > datetime.utcnow()
                ).all()
                
                for key in active_keys:
                    self.keypairs[key.key_id] = {
                        'key_id': key.key_id,
                        'public_key': serialization.load_pem_public_key(key.public_key.encode()),
                        'private_key': serialization.load_pem_private_key(
                            key.private_key.encode(),
                            password=None
                        ),
                        'algorithm': key.algorithm,
                        'expires_at': key.expires_at
                    }
                    
                # Set current keypair (most recent)
                if active_keys:
                    current_key = max(active_keys, key=lambda k: k.created_at)
                    self.current_keypair = self.keypairs[current_key.key_id]
                else:
                    # Generate first keypair
                    await self._generate_new_keypair()
                    
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"JWT key initialization error: {e}")
            raise
            
    async def _generate_new_keypair(self) -> str:
        """Generate new RSA keypair for JWT signing"""
        try:
            # Generate RSA key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            public_key = private_key.public_key()
            
            # Serialize keys
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            # Generate key ID
            key_id = f"key_{int(time.time())}"
            expires_at = datetime.utcnow() + timedelta(days=30)
            
            # Store in database
            db = self.SessionLocal()
            try:
                jwt_key = JWTKeyPair(
                    key_id=key_id,
                    public_key=public_pem.decode(),
                    private_key=private_pem.decode(),
                    algorithm="RS256",
                    expires_at=expires_at
                )
                db.add(jwt_key)
                db.commit()
                
                # Add to memory
                self.keypairs[key_id] = {
                    'key_id': key_id,
                    'public_key': public_key,
                    'private_key': private_key,
                    'algorithm': 'RS256',
                    'expires_at': expires_at
                }
                
                # Set as current if no current keypair
                if not self.current_keypair:
                    self.current_keypair = self.keypairs[key_id]
                    
                logger.info(f"Generated new JWT keypair: {key_id}")
                return key_id
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Keypair generation error: {e}")
            raise
            
    def _initialize_rbac(self) -> Dict[Role, Set[Permission]]:
        """Initialize role-based access control permissions"""
        return {
            Role.SUPER_ADMIN: {
                Permission.CHAT_READ, Permission.CHAT_WRITE, Permission.CHAT_DELETE, Permission.CHAT_ADMIN,
                Permission.USER_READ, Permission.USER_WRITE, Permission.USER_DELETE, Permission.USER_ADMIN,
                Permission.ANALYTICS_READ, Permission.ANALYTICS_WRITE, Permission.ANALYTICS_ADMIN,
                Permission.SYSTEM_ADMIN, Permission.SYSTEM_CONFIG, Permission.SYSTEM_MONITOR,
                Permission.BILLING_READ, Permission.BILLING_WRITE, Permission.BILLING_ADMIN
            },
            Role.ADMIN: {
                Permission.CHAT_READ, Permission.CHAT_WRITE, Permission.CHAT_DELETE,
                Permission.USER_READ, Permission.USER_WRITE,
                Permission.ANALYTICS_READ, Permission.ANALYTICS_WRITE,
                Permission.SYSTEM_MONITOR,
                Permission.BILLING_READ, Permission.BILLING_WRITE
            },
            Role.MANAGER: {
                Permission.CHAT_READ, Permission.CHAT_WRITE,
                Permission.USER_READ,
                Permission.ANALYTICS_READ,
                Permission.BILLING_READ
            },
            Role.USER: {
                Permission.CHAT_READ, Permission.CHAT_WRITE,
                Permission.USER_READ
            },
            Role.VIEWER: {
                Permission.CHAT_READ,
                Permission.USER_READ,
                Permission.ANALYTICS_READ
            },
            Role.SERVICE: {
                Permission.CHAT_READ, Permission.CHAT_WRITE,
                Permission.ANALYTICS_READ
            }
        }
        
    def _setup_oauth_providers(self):
        """Setup OAuth2 providers"""
        oauth_config = self.config.get('oauth_providers', {})
        
        # Google OAuth2
        if 'google' in oauth_config:
            self.oauth.register(
                name='google',
                client_id=oauth_config['google']['client_id'],
                client_secret=oauth_config['google']['client_secret'],
                server_metadata_url='https://accounts.google.com/.well-known/openid_configuration',
                client_kwargs={
                    'scope': 'openid email profile'
                }
            )
            
        # Microsoft Azure AD
        if 'azure' in oauth_config:
            tenant_id = oauth_config['azure']['tenant_id']
            self.oauth.register(
                name='azure',
                client_id=oauth_config['azure']['client_id'],
                client_secret=oauth_config['azure']['client_secret'],
                authorization_endpoint=f'https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/authorize',
                token_endpoint=f'https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token',
                client_kwargs={
                    'scope': 'openid profile email'
                }
            )
            
    def _setup_saml_providers(self):
        """Setup SAML SSO providers"""
        saml_config = self.config.get('saml_providers', {})
        
        for provider_name, provider_config in saml_config.items():
            try:
                config = Saml2Config()
                config.load({
                    'entityid': provider_config['entity_id'],
                    'description': f'SAML SSO for {provider_name}',
                    'service': {
                        'sp': {
                            'name': 'AI Chatbot Service Provider',
                            'endpoints': {
                                'assertion_consumer_service': [
                                    (provider_config['acs_url'], BINDING_HTTP_POST),
                                ],
                                'single_logout_service': [
                                    (provider_config['sls_url'], BINDING_HTTP_REDIRECT),
                                ],
                            },
                            'allow_unsolicited': True,
                            'authn_requests_signed': False,
                            'logout_requests_signed': True,
                            'want_assertions_signed': True,
                            'want_response_signed': False,
                        },
                    },
                    'metadata': {
                        'remote': [
                            {
                                'url': provider_config['metadata_url'],
                            },
                        ],
                    },
                    'key_file': provider_config.get('private_key_file'),
                    'cert_file': provider_config.get('certificate_file'),
                })
                
                self.saml_clients[provider_name] = Saml2Client(config)
                
            except Exception as e:
                logger.error(f"SAML provider setup error for {provider_name}: {e}")
                
    def _setup_ldap_servers(self):
        """Setup LDAP servers for authentication"""
        ldap_config = self.config.get('ldap_servers', {})
        
        for server_name, server_config in ldap_config.items():
            self.ldap_servers[server_name] = {
                'server': Server(
                    server_config['host'],
                    port=server_config.get('port', 389),
                    use_ssl=server_config.get('use_ssl', False),
                    get_info=ALL
                ),
                'base_dn': server_config['base_dn'],
                'user_filter': server_config.get('user_filter', '(uid={username})'),
                'bind_dn': server_config.get('bind_dn'),
                'bind_password': server_config.get('bind_password')
            }
            
    async def authenticate_user(self, email: str, password: str, mfa_code: Optional[str] = None, request_info: Dict[str, Any] = None) -> Optional[AuthContext]:
        """Authenticate user with email/password and optional MFA"""
        try:
            # Rate limiting check
            if not await self._check_rate_limit('login', email, request_info):
                raise HTTPException(status_code=429, detail="Too many login attempts")
                
            db = self.SessionLocal()
            try:
                # Find user
                user = db.query(User).filter(User.email == email).first()
                if not user or not user.is_active:
                    await self._log_auth_event(None, 'login_failed', 'user_not_found', request_info, False)
                    return None
                    
                # Check account lockout
                if user.locked_until and user.locked_until > datetime.utcnow():
                    await self._log_auth_event(user.id, 'login_failed', 'account_locked', request_info, False)
                    return None
                    
                # Verify password
                if not user.password_hash or not bcrypt.checkpw(password.encode(), user.password_hash.encode()):
                    # Increment failed attempts
                    user.failed_login_attempts += 1
                    if user.failed_login_attempts >= 5:
                        user.locked_until = datetime.utcnow() + timedelta(minutes=30)
                    db.commit()
                    
                    await self._log_auth_event(user.id, 'login_failed', 'invalid_password', request_info, False)
                    return None
                    
                # Check MFA if enabled
                mfa_verified = True
                if user.mfa_enabled:
                    if not mfa_code:
                        await self._log_auth_event(user.id, 'login_failed', 'mfa_required', request_info, False)
                        raise HTTPException(status_code=200, detail="MFA required", headers={"X-MFA-Required": "true"})
                        
                    if not await self._verify_mfa(user, mfa_code):
                        await self._log_auth_event(user.id, 'login_failed', 'invalid_mfa', request_info, False)
                        return None
                        
                # Reset failed attempts on successful login
                user.failed_login_attempts = 0
                user.locked_until = None
                user.last_login = datetime.utcnow()
                db.commit()
                
                # Get user roles and permissions
                roles = await self._get_user_roles(user.id)
                permissions = await self._get_user_permissions(roles)
                
                # Create auth context
                auth_context = AuthContext(
                    user_id=str(user.id),
                    email=user.email,
                    roles=[role.value for role in roles],
                    permissions=permissions,
                    session_id="",  # Will be set when session is created
                    auth_method=AuthMethod.PASSWORD,
                    mfa_verified=mfa_verified,
                    expires_at=datetime.utcnow() + timedelta(seconds=self.access_token_expire)
                )
                
                await self._log_auth_event(user.id, 'login_success', 'password_auth', request_info, True)
                return auth_context
                
            finally:
                db.close()
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
            
    async def create_session(self, auth_context: AuthContext, request_info: Dict[str, Any] = None) -> TokenResponse:
        """Create user session with JWT tokens"""
        try:
            session_id = str(uuid.uuid4())
            refresh_token = secrets.token_urlsafe(32)
            
            # Create session record
            db = self.SessionLocal()
            try:
                session = UserSession(
                    id=uuid.UUID(session_id),
                    user_id=uuid.UUID(auth_context.user_id),
                    session_token=session_id,
                    refresh_token=refresh_token,
                    device_info=request_info.get('device_info', {}),
                    ip_address=request_info.get('ip_address'),
                    user_agent=request_info.get('user_agent'),
                    expires_at=datetime.utcnow() + timedelta(seconds=self.refresh_token_expire)
                )
                db.add(session)
                db.commit()
                
            finally:
                db.close()
                
            # Update auth context with session
            auth_context.session_id = session_id
            
            # Generate JWT tokens
            access_token = await self._generate_access_token(auth_context)
            
            # Store session in Redis
            await self.redis.setex(
                f"session:{session_id}",
                self.access_token_expire,
                json.dumps({
                    'user_id': auth_context.user_id,
                    'email': auth_context.email,
                    'roles': auth_context.roles,
                    'permissions': [p.value for p in auth_context.permissions],
                    'mfa_verified': auth_context.mfa_verified
                })
            )
            
            return TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                token_type="bearer",
                expires_in=self.access_token_expire,
                scope=" ".join([p.value for p in auth_context.permissions])
            )
            
        except Exception as e:
            logger.error(f"Session creation error: {e}")
            raise
            
    async def _generate_access_token(self, auth_context: AuthContext) -> str:
        """Generate JWT access token"""
        try:
            if not self.current_keypair:
                raise Exception("No active JWT keypair available")
                
            now = datetime.utcnow()
            payload = {
                'iss': self.jwt_issuer,
                'aud': self.jwt_audience,
                'sub': auth_context.user_id,
                'email': auth_context.email,
                'roles': auth_context.roles,
                'permissions': [p.value for p in auth_context.permissions],
                'session_id': auth_context.session_id,
                'auth_method': auth_context.auth_method.value,
                'mfa_verified': auth_context.mfa_verified,
                'iat': int(now.timestamp()),
                'exp': int(auth_context.expires_at.timestamp()),
                'jti': str(uuid.uuid4())
            }
            
            # Sign with current private key
            token = jwt.encode(
                payload,
                self.current_keypair['private_key'],
                algorithm=self.current_keypair['algorithm'],
                headers={'kid': self.current_keypair['key_id']}
            )
            
            return token
            
        except Exception as e:
            logger.error(f"Token generation error: {e}")
            raise
            
    async def verify_token(self, token: str) -> Optional[AuthContext]:
        """Verify JWT token and return auth context"""
        try:
            # Decode header to get key ID
            unverified_header = jwt.get_unverified_header(token)
            key_id = unverified_header.get('kid')
            
            if not key_id or key_id not in self.keypairs:
                return None
                
            # Verify token
            keypair = self.keypairs[key_id]
            payload = jwt.decode(
                token,
                keypair['public_key'],
                algorithms=[keypair['algorithm']],
                audience=self.jwt_audience,
                issuer=self.jwt_issuer
            )
            
            # Check session validity
            session_id = payload.get('session_id')
            if session_id:
                session_data = await self.redis.get(f"session:{session_id}")
                if not session_data:
                    return None
                    
            # Create auth context
            permissions = {Permission(p) for p in payload.get('permissions', [])}
            
            return AuthContext(
                user_id=payload['sub'],
                email=payload['email'],
                roles=payload.get('roles', []),
                permissions=permissions,
                session_id=session_id,
                auth_method=AuthMethod(payload.get('auth_method', 'password')),
                mfa_verified=payload.get('mfa_verified', False),
                expires_at=datetime.fromtimestamp(payload['exp'])
            )
            
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None
            
    async def refresh_token(self, refresh_token: str, request_info: Dict[str, Any] = None) -> Optional[TokenResponse]:
        """Refresh access token using refresh token"""
        try:
            db = self.SessionLocal()
            try:
                # Find session by refresh token
                session = db.query(UserSession).filter(
                    UserSession.refresh_token == refresh_token,
                    UserSession.is_active == True,
                    UserSession.expires_at > datetime.utcnow()
                ).first()
                
                if not session:
                    return None
                    
                # Get user
                user = db.query(User).filter(User.id == session.user_id).first()
                if not user or not user.is_active:
                    return None
                    
                # Update session activity
                session.last_activity = datetime.utcnow()
                db.commit()
                
                # Get roles and permissions
                roles = await self._get_user_roles(session.user_id)
                permissions = await self._get_user_permissions(roles)
                
                # Create new auth context
                auth_context = AuthContext(
                    user_id=str(session.user_id),
                    email=user.email,
                    roles=[role.value for role in roles],
                    permissions=permissions,
                    session_id=str(session.id),
                    auth_method=AuthMethod.PASSWORD,  # Could be stored in session
                    mfa_verified=True,  # Assume MFA verified for existing session
                    expires_at=datetime.utcnow() + timedelta(seconds=self.access_token_expire)
                )
                
                # Generate new access token
                access_token = await self._generate_access_token(auth_context)
                
                # Update Redis session
                await self.redis.setex(
                    f"session:{session.id}",
                    self.access_token_expire,
                    json.dumps({
                        'user_id': auth_context.user_id,
                        'email': auth_context.email,
                        'roles': auth_context.roles,
                        'permissions': [p.value for p in auth_context.permissions],
                        'mfa_verified': auth_context.mfa_verified
                    })
                )
                
                return TokenResponse(
                    access_token=access_token,
                    refresh_token=refresh_token,
                    token_type="bearer",
                    expires_in=self.access_token_expire,
                    scope=" ".join([p.value for p in auth_context.permissions])
                )
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return None
            
    async def setup_mfa(self, user_id: str, method: str, phone_number: Optional[str] = None) -> Dict[str, Any]:
        """Setup multi-factor authentication for user"""
        try:
            db = self.SessionLocal()
            try:
                user = db.query(User).filter(User.id == uuid.UUID(user_id)).first()
                if not user:
                    raise HTTPException(status_code=404, detail="User not found")
                    
                if method == "totp":
                    # Generate TOTP secret
                    secret = pyotp.random_base32()
                    
                    # Create provisioning URI
                    totp = pyotp.TOTP(secret)
                    provisioning_uri = totp.provisioning_uri(
                        name=user.email,
                        issuer_name="AI Chatbot System"
                    )
                    
                    # Generate QR code
                    qr = qrcode.QRCode(version=1, box_size=10, border=5)
                    qr.add_data(provisioning_uri)
                    qr.make(fit=True)
                    
                    img = qr.make_image(fill_color="black", back_color="white")
                    buffer = BytesIO()
                    img.save(buffer, format='PNG')
                    qr_code_data = base64.b64encode(buffer.getvalue()).decode()
                    
                    # Store secret (will be activated after verification)
                    user.mfa_secret = secret
                    db.commit()
                    
                    return {
                        'method': 'totp',
                        'secret': secret,
                        'qr_code': qr_code_data,
                        'provisioning_uri': provisioning_uri
                    }
                    
                elif method == "sms":
                    if not phone_number:
                        raise HTTPException(status_code=400, detail="Phone number required for SMS MFA")
                        
                    # Store phone number
                    user.phone_number = phone_number
                    db.commit()
                    
                    # Send verification SMS (implementation depends on SMS provider)
                    verification_code = f"{secrets.randbelow(900000) + 100000:06d}"
                    await self._send_sms_verification(phone_number, verification_code)
                    
                    # Store verification code temporarily
                    await self.redis.setex(
                        f"mfa_verification:{user_id}",
                        300,  # 5 minutes
                        verification_code
                    )
                    
                    return {
                        'method': 'sms',
                        'phone_number': phone_number,
                        'message': 'Verification code sent to phone'
                    }
                    
                else:
                    raise HTTPException(status_code=400, detail="Invalid MFA method")
                    
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"MFA setup error: {e}")
            raise
            
    async def verify_mfa_setup(self, user_id: str, code: str) -> bool:
        """Verify MFA setup with provided code"""
        try:
            db = self.SessionLocal()
            try:
                user = db.query(User).filter(User.id == uuid.UUID(user_id)).first()
                if not user:
                    return False
                    
                if user.mfa_secret:
                    # TOTP verification
                    totp = pyotp.TOTP(user.mfa_secret)
                    if totp.verify(code, valid_window=1):
                        user.mfa_enabled = True
                        db.commit()
                        return True
                        
                elif user.phone_number:
                    # SMS verification
                    stored_code = await self.redis.get(f"mfa_verification:{user_id}")
                    if stored_code and stored_code.decode() == code:
                        user.mfa_enabled = True
                        db.commit()
                        await self.redis.delete(f"mfa_verification:{user_id}")
                        return True
                        
                return False
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"MFA verification error: {e}")
            return False
            
    async def _verify_mfa(self, user: User, code: str) -> bool:
        """Verify MFA code during login"""
        try:
            if user.mfa_secret:
                # TOTP verification
                totp = pyotp.TOTP(user.mfa_secret)
                return totp.verify(code, valid_window=1)
                
            elif user.phone_number:
                # SMS verification (would need to be sent first)
                # This is a simplified implementation
                stored_code = await self.redis.get(f"mfa_login:{user.id}")
                if stored_code and stored_code.decode() == code:
                    await self.redis.delete(f"mfa_login:{user.id}")
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"MFA verification error: {e}")
            return False
            
    async def authenticate_service_mesh(self, service_cert: str, service_name: str) -> Optional[AuthContext]:
        """Authenticate service-to-service requests in service mesh"""
        try:
            # Verify service certificate with Consul
            service_info = self.consul_client.catalog.service(service_name)
            if not service_info[1]:  # No service found
                return None
                
            # In a real implementation, you would verify the mutual TLS certificate
            # For now, we'll create a service auth context
            
            service_permissions = {
                Permission.CHAT_READ,
                Permission.CHAT_WRITE,
                Permission.ANALYTICS_READ
            }
            
            return AuthContext(
                user_id=f"service:{service_name}",
                email=f"{service_name}@service.local",
                roles=[Role.SERVICE.value],
                permissions=service_permissions,
                session_id=str(uuid.uuid4()),
                auth_method=AuthMethod.SERVICE_MESH,
                mfa_verified=True,  # Services don't need MFA
                expires_at=datetime.utcnow() + timedelta(hours=1)
            )
            
        except Exception as e:
            logger.error(f"Service mesh authentication error: {e}")
            return None
            
    async def authenticate_ldap(self, username: str, password: str, server_name: str = "default") -> Optional[AuthContext]:
        """Authenticate user against LDAP server"""
        try:
            if server_name not in self.ldap_servers:
                return None
                
            ldap_config = self.ldap_servers[server_name]
            
            # Create LDAP connection
            conn = Connection(
                ldap_config['server'],
                user=ldap_config.get('bind_dn'),
                password=ldap_config.get('bind_password'),
                auto_bind=True
            )
            
            # Search for user
            user_filter = ldap_config['user_filter'].format(username=username)
            conn.search(
                ldap_config['base_dn'],
                user_filter,
                SUBTREE,
                attributes=['mail', 'cn', 'memberOf']
            )
            
            if not conn.entries:
                return None
                
            user_entry = conn.entries[0]
            user_dn = user_entry.entry_dn
            
            # Authenticate user
            auth_conn = Connection(
                ldap_config['server'],
                user=user_dn,
                password=password,
                auto_bind=True
            )
            
            if not auth_conn.bind():
                return None
                
            # Extract user information
            email = str(user_entry.mail) if user_entry.mail else f"{username}@ldap.local"
            display_name = str(user_entry.cn) if user_entry.cn else username
            
            # Map LDAP groups to roles (simplified)
            roles = [Role.USER]  # Default role
            if user_entry.memberOf:
                groups = [str(group) for group in user_entry.memberOf]
                if any('admin' in group.lower() for group in groups):
                    roles.append(Role.ADMIN)
                    
            permissions = set()
            for role in roles:
                permissions.update(self.role_permissions.get(role, set()))
                
            return AuthContext(
                user_id=f"ldap:{username}",
                email=email,
                roles=[role.value for role in roles],
                permissions=permissions,
                session_id="",
                auth_method=AuthMethod.LDAP,
                mfa_verified=False,
                expires_at=datetime.utcnow() + timedelta(seconds=self.access_token_expire)
            )
            
        except Exception as e:
            logger.error(f"LDAP authentication error: {e}")
            return None
            
    async def has_permission(self, auth_context: AuthContext, permission: Permission, resource_type: Optional[str] = None, resource_id: Optional[str] = None) -> bool:
        """Check if user has specific permission"""
        try:
            # Check direct permission
            if permission in auth_context.permissions:
                return True
                
            # Check resource-specific permissions
            if resource_type and resource_id:
                db = self.SessionLocal()
                try:
                    resource_role = db.query(UserRole).filter(
                        UserRole.user_id == uuid.UUID(auth_context.user_id),
                        UserRole.resource_type == resource_type,
                        UserRole.resource_id == resource_id,
                        UserRole.expires_at.is_(None) or UserRole.expires_at > datetime.utcnow()
                    ).first()
                    
                    if resource_role:
                        role_enum = Role(resource_role.role)
                        role_permissions = self.role_permissions.get(role_enum, set())
                        return permission in role_permissions
                        
                finally:
                    db.close()
                    
            return False
            
        except Exception as e:
            logger.error(f"Permission check error: {e}")
            return False
            
    async def assign_role(self, user_id: str, role: Role, granted_by: str, resource_type: Optional[str] = None, resource_id: Optional[str] = None, expires_at: Optional[datetime] = None) -> bool:
        """Assign role to user"""
        try:
            db = self.SessionLocal()
            try:
                role_assignment = UserRole(
                    user_id=uuid.UUID(user_id),
                    role=role.value,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    granted_by=uuid.UUID(granted_by),
                    expires_at=expires_at
                )
                db.add(role_assignment)
                db.commit()
                
                return True
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Role assignment error: {e}")
            return False
            
    async def _get_user_roles(self, user_id: uuid.UUID) -> List[Role]:
        """Get user's roles"""
        try:
            db = self.SessionLocal()
            try:
                role_assignments = db.query(UserRole).filter(
                    UserRole.user_id == user_id,
                    UserRole.expires_at.is_(None) or UserRole.expires_at > datetime.utcnow()
                ).all()
                
                roles = []
                for assignment in role_assignments:
                    try:
                        roles.append(Role(assignment.role))
                    except ValueError:
                        pass  # Skip invalid roles
                        
                return roles if roles else [Role.USER]  # Default role
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Get user roles error: {e}")
            return [Role.USER]
            
    async def _get_user_permissions(self, roles: List[Role]) -> Set[Permission]:
        """Get permissions for user roles"""
        permissions = set()
        for role in roles:
            permissions.update(self.role_permissions.get(role, set()))
        return permissions
        
    async def _check_rate_limit(self, action: str, identifier: str, request_info: Dict[str, Any] = None) -> bool:
        """Check rate limit for action"""
        try:
            if action not in self.rate_limits:
                return True
                
            limit_config = self.rate_limits[action]
            key = f"rate_limit:{action}:{identifier}"
            
            current_time = time.time()
            window_start = current_time - limit_config['window']
            
            # Clean old entries and count current requests
            pipe = self.redis.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zcard(key)
            pipe.expire(key, limit_config['window'])
            results = await pipe.execute()
            
            current_count = results[1]
            
            if current_count >= limit_config['requests']:
                return False
                
            # Add current request
            await self.redis.zadd(key, {str(uuid.uuid4()): current_time})
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return True  # Allow on error
            
    async def _log_auth_event(self, user_id: Optional[uuid.UUID], action: str, resource: str, request_info: Dict[str, Any] = None, success: bool = True):
        """Log authentication event"""
        try:
            db = self.SessionLocal()
            try:
                audit_log = AuditLog(
                    user_id=user_id,
                    action=action,
                    resource=resource,
                    ip_address=request_info.get('ip_address') if request_info else None,
                    user_agent=request_info.get('user_agent') if request_info else None,
                    success=success,
                    details=request_info or {}
                )
                db.add(audit_log)
                db.commit()
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Audit logging error: {e}")
            
    async def _send_sms_verification(self, phone_number: str, code: str):
        """Send SMS verification code"""
        # This would integrate with SMS provider like Twilio
        # For now, just log the code
        logger.info(f"SMS verification code for {phone_number}: {code}")
        
    async def _key_rotation_task(self):
        """Background task for JWT key rotation"""
        while True:
            try:
                await asyncio.sleep(86400)  # Check daily
                
                if self.current_keypair:
                    expires_at = self.current_keypair['expires_at']
                    if expires_at and expires_at < datetime.utcnow() + timedelta(days=7):
                        # Generate new keypair before current expires
                        await self._generate_new_keypair()
                        
                        # Clean up expired keypairs
                        expired_keys = []
                        for key_id, keypair in self.keypairs.items():
                            if keypair['expires_at'] < datetime.utcnow():
                                expired_keys.append(key_id)
                                
                        for key_id in expired_keys:
                            del self.keypairs[key_id]
                            
                        logger.info(f"Cleaned up {len(expired_keys)} expired keypairs")
                        
            except Exception as e:
                logger.error(f"Key rotation task error: {e}")
                
    async def _session_cleanup_task(self):
        """Background task for cleaning up expired sessions"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run hourly
                
                db = self.SessionLocal()
                try:
                    # Mark expired sessions as inactive
                    expired_count = db.query(UserSession).filter(
                        UserSession.expires_at < datetime.utcnow(),
                        UserSession.is_active == True
                    ).update({'is_active': False})
                    
                    db.commit()
                    
                    if expired_count > 0:
                        logger.info(f"Cleaned up {expired_count} expired sessions")
                        
                finally:
                    db.close()
                    
            except Exception as e:
                logger.error(f"Session cleanup task error: {e}")
                
    async def _audit_cleanup_task(self):
        """Background task for cleaning up old audit logs"""
        while True:
            try:
                await asyncio.sleep(86400)  # Run daily
                
                cutoff_date = datetime.utcnow() - timedelta(days=365)  # Keep 1 year
                
                db = self.SessionLocal()
                try:
                    deleted_count = db.query(AuditLog).filter(
                        AuditLog.timestamp < cutoff_date
                    ).delete()
                    
                    db.commit()
                    
                    if deleted_count > 0:
                        logger.info(f"Cleaned up {deleted_count} old audit logs")
                        
                finally:
                    db.close()
                    
            except Exception as e:
                logger.error(f"Audit cleanup task error: {e}")
                
    def get_jwks(self) -> Dict[str, Any]:
        """Get JSON Web Key Set for token verification"""
        try:
            keys = []
            
            for key_id, keypair in self.keypairs.items():
                if keypair['expires_at'] > datetime.utcnow():
                    public_key = keypair['public_key']
                    
                    # Convert to JWK format
                    jwk = RSAAlgorithm.to_jwk(public_key)
                    jwk_dict = json.loads(jwk)
                    jwk_dict['kid'] = key_id
                    jwk_dict['use'] = 'sig'
                    jwk_dict['alg'] = keypair['algorithm']
                    
                    keys.append(jwk_dict)
                    
            return {'keys': keys}
            
        except Exception as e:
            logger.error(f"JWKS generation error: {e}")
            return {'keys': []}
            
    async def logout(self, session_id: str):
        """Logout user and invalidate session"""
        try:
            # Remove from Redis
            await self.redis.delete(f"session:{session_id}")
            
            # Mark session as inactive in database
            db = self.SessionLocal()
            try:
                db.query(UserSession).filter(
                    UserSession.id == uuid.UUID(session_id)
                ).update({'is_active': False})
                db.commit()
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Logout error: {e}")
            
    async def shutdown(self):
        """Shutdown the authentication service"""
        try:
            await self.redis.close()
            logger.info("Unified Authentication Service shut down")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")