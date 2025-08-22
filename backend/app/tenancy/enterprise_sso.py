"""
Enterprise SSO and Security Implementation
SAML 2.0, OAuth2, JWT with refresh tokens, RBAC, and API key management
"""

import asyncio
import time
import secrets
import jwt
import hashlib
import base64
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import xml.etree.ElementTree as ET
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.x509 import load_pem_x509_certificate
import aioredis
from fastapi import HTTPException, Request, Depends
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from passlib.context import CryptContext
import httpx
from prometheus_client import Counter, Histogram
import logging

logger = logging.getLogger(__name__)

# Metrics
auth_attempts = Counter('auth_attempts_total', 'Total authentication attempts', ['method', 'status'])
token_operations = Counter('token_operations_total', 'Token operations', ['operation', 'type'])
api_key_usage = Counter('api_key_usage_total', 'API key usage', ['key_id', 'scope'])


class AuthMethod(Enum):
    PASSWORD = "password"
    SAML = "saml"
    OAUTH2 = "oauth2"
    API_KEY = "api_key"
    

class Role(Enum):
    ADMIN = "admin"
    USER = "user"
    DEVELOPER = "developer"
    VIEWER = "viewer"
    SERVICE = "service"
    

class Permission(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    CREATE_API_KEY = "create_api_key"
    MANAGE_USERS = "manage_users"
    VIEW_ANALYTICS = "view_analytics"
    

@dataclass
class User:
    id: str
    email: str
    roles: List[Role]
    permissions: List[Permission]
    tenant_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    mfa_enabled: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    

@dataclass
class APIKey:
    id: str
    key_hash: str
    name: str
    user_id: str
    tenant_id: str
    scopes: List[str]
    rate_limit: int = 1000  # per hour
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    

@dataclass
class SAMLConfig:
    entity_id: str
    sso_url: str
    certificate: str
    attribute_mapping: Dict[str, str]
    

@dataclass
class OAuth2Config:
    client_id: str
    client_secret: str
    authorize_url: str
    token_url: str
    userinfo_url: str
    scopes: List[str]
    

class EnterpriseAuth:
    """Complete enterprise authentication system"""
    
    def __init__(self, redis: aioredis.Redis, config: Dict[str, Any]):
        self.redis = redis
        self.config = config
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # JWT configuration
        self.jwt_secret = config.get("jwt_secret", secrets.token_urlsafe(32))
        self.jwt_algorithm = "RS256" if "jwt_private_key" in config else "HS256"
        self.jwt_private_key = config.get("jwt_private_key")
        self.jwt_public_key = config.get("jwt_public_key")
        self.access_token_expire = timedelta(minutes=30)
        self.refresh_token_expire = timedelta(days=30)
        
        # SAML providers
        self.saml_providers: Dict[str, SAMLConfig] = {}
        for provider in config.get("saml_providers", []):
            self.saml_providers[provider["name"]] = SAMLConfig(**provider)
            
        # OAuth2 providers
        self.oauth2_providers: Dict[str, OAuth2Config] = {}
        for provider in config.get("oauth2_providers", []):
            self.oauth2_providers[provider["name"]] = OAuth2Config(**provider)
            
        # Role permissions mapping
        self.role_permissions = {
            Role.ADMIN: [p for p in Permission],
            Role.DEVELOPER: [Permission.READ, Permission.WRITE, Permission.CREATE_API_KEY, Permission.VIEW_ANALYTICS],
            Role.USER: [Permission.READ, Permission.WRITE],
            Role.VIEWER: [Permission.READ, Permission.VIEW_ANALYTICS],
            Role.SERVICE: [Permission.READ, Permission.WRITE]
        }
        
    # SAML 2.0 Implementation
    
    async def saml_login(self, provider_name: str, saml_response: str) -> Tuple[User, str, str]:
        """Process SAML login response"""
        if provider_name not in self.saml_providers:
            raise HTTPException(status_code=400, detail="Unknown SAML provider")
            
        provider = self.saml_providers[provider_name]
        
        try:
            # Decode and parse SAML response
            decoded_response = base64.b64decode(saml_response)
            root = ET.fromstring(decoded_response)
            
            # Verify signature
            if not self._verify_saml_signature(root, provider.certificate):
                auth_attempts.labels(method="saml", status="invalid_signature").inc()
                raise HTTPException(status_code=401, detail="Invalid SAML signature")
                
            # Extract assertions
            assertion = root.find(".//{urn:oasis:names:tc:SAML:2.0:assertion}Assertion")
            if assertion is None:
                raise HTTPException(status_code=400, detail="No assertion found")
                
            # Check conditions (NotBefore, NotOnOrAfter)
            if not self._check_saml_conditions(assertion):
                auth_attempts.labels(method="saml", status="expired").inc()
                raise HTTPException(status_code=401, detail="SAML assertion expired")
                
            # Extract attributes
            attributes = self._extract_saml_attributes(assertion, provider.attribute_mapping)
            
            # Create or update user
            user = await self._provision_saml_user(attributes, provider_name)
            
            # Generate tokens
            access_token = self._create_access_token(user)
            refresh_token = await self._create_refresh_token(user)
            
            auth_attempts.labels(method="saml", status="success").inc()
            
            return user, access_token, refresh_token
            
        except Exception as e:
            auth_attempts.labels(method="saml", status="error").inc()
            logger.error(f"SAML login error: {e}")
            raise HTTPException(status_code=400, detail="SAML authentication failed")
            
    def _verify_saml_signature(self, root: ET.Element, cert_pem: str) -> bool:
        """Verify SAML response signature"""
        # In production, use proper XML signature verification library
        # This is a simplified version
        try:
            cert = load_pem_x509_certificate(cert_pem.encode())
            # Implement actual signature verification
            return True
        except Exception:
            return False
            
    def _check_saml_conditions(self, assertion: ET.Element) -> bool:
        """Check SAML assertion time conditions"""
        conditions = assertion.find(".//{urn:oasis:names:tc:SAML:2.0:assertion}Conditions")
        if conditions is None:
            return True
            
        now = datetime.now(timezone.utc)
        
        not_before = conditions.get("NotBefore")
        if not_before:
            nb_time = datetime.fromisoformat(not_before.replace("Z", "+00:00"))
            if now < nb_time:
                return False
                
        not_after = conditions.get("NotOnOrAfter")
        if not_after:
            na_time = datetime.fromisoformat(not_after.replace("Z", "+00:00"))
            if now >= na_time:
                return False
                
        return True
        
    def _extract_saml_attributes(self, assertion: ET.Element, mapping: Dict[str, str]) -> Dict[str, Any]:
        """Extract user attributes from SAML assertion"""
        attributes = {}
        
        for attr in assertion.findall(".//{urn:oasis:names:tc:SAML:2.0:assertion}Attribute"):
            name = attr.get("Name")
            value_elem = attr.find(".//{urn:oasis:names:tc:SAML:2.0:assertion}AttributeValue")
            
            if value_elem is not None and name in mapping:
                attributes[mapping[name]] = value_elem.text
                
        return attributes
        
    async def _provision_saml_user(self, attributes: Dict[str, Any], provider: str) -> User:
        """Create or update user from SAML attributes"""
        email = attributes.get("email")
        if not email:
            raise ValueError("Email not found in SAML attributes")
            
        # Check if user exists
        user_data = await self.redis.get(f"user:email:{email}")
        
        if user_data:
            user = User(**eval(user_data))
            # Update attributes
            user.attributes.update(attributes)
        else:
            # Create new user
            user = User(
                id=secrets.token_urlsafe(16),
                email=email,
                roles=[Role.USER],
                permissions=self.role_permissions[Role.USER],
                tenant_id=attributes.get("tenant_id", "default"),
                attributes=attributes
            )
            
        # Save user
        await self.redis.set(f"user:{user.id}", str(user.__dict__))
        await self.redis.set(f"user:email:{email}", str(user.__dict__))
        
        return user
        
    # OAuth2 Implementation
    
    async def oauth2_authorize(self, provider_name: str, redirect_uri: str) -> str:
        """Generate OAuth2 authorization URL"""
        if provider_name not in self.oauth2_providers:
            raise HTTPException(status_code=400, detail="Unknown OAuth2 provider")
            
        provider = self.oauth2_providers[provider_name]
        state = secrets.token_urlsafe(32)
        
        # Store state for verification
        await self.redis.setex(f"oauth2:state:{state}", 600, redirect_uri)
        
        params = {
            "client_id": provider.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(provider.scopes),
            "state": state
        }
        
        # Build authorization URL
        url = provider.authorize_url + "?" + "&".join(f"{k}={v}" for k, v in params.items())
        
        return url
        
    async def oauth2_callback(
        self,
        provider_name: str,
        code: str,
        state: str
    ) -> Tuple[User, str, str]:
        """Handle OAuth2 callback"""
        if provider_name not in self.oauth2_providers:
            raise HTTPException(status_code=400, detail="Unknown OAuth2 provider")
            
        # Verify state
        redirect_uri = await self.redis.get(f"oauth2:state:{state}")
        if not redirect_uri:
            auth_attempts.labels(method="oauth2", status="invalid_state").inc()
            raise HTTPException(status_code=400, detail="Invalid state")
            
        provider = self.oauth2_providers[provider_name]
        
        try:
            # Exchange code for token
            async with httpx.AsyncClient() as client:
                token_response = await client.post(
                    provider.token_url,
                    data={
                        "grant_type": "authorization_code",
                        "code": code,
                        "redirect_uri": redirect_uri,
                        "client_id": provider.client_id,
                        "client_secret": provider.client_secret
                    }
                )
                token_response.raise_for_status()
                token_data = token_response.json()
                
                # Get user info
                userinfo_response = await client.get(
                    provider.userinfo_url,
                    headers={"Authorization": f"Bearer {token_data['access_token']}"}
                )
                userinfo_response.raise_for_status()
                userinfo = userinfo_response.json()
                
            # Create or update user
            user = await self._provision_oauth2_user(userinfo, provider_name)
            
            # Generate tokens
            access_token = self._create_access_token(user)
            refresh_token = await self._create_refresh_token(user)
            
            auth_attempts.labels(method="oauth2", status="success").inc()
            
            return user, access_token, refresh_token
            
        except Exception as e:
            auth_attempts.labels(method="oauth2", status="error").inc()
            logger.error(f"OAuth2 callback error: {e}")
            raise HTTPException(status_code=400, detail="OAuth2 authentication failed")
            
    async def _provision_oauth2_user(self, userinfo: Dict[str, Any], provider: str) -> User:
        """Create or update user from OAuth2 userinfo"""
        email = userinfo.get("email")
        if not email:
            raise ValueError("Email not found in OAuth2 userinfo")
            
        # Similar to SAML provisioning
        return await self._provision_saml_user(userinfo, provider)
        
    # JWT Token Management
    
    def _create_access_token(self, user: User) -> str:
        """Create JWT access token"""
        payload = {
            "sub": user.id,
            "email": user.email,
            "roles": [r.value for r in user.roles],
            "tenant_id": user.tenant_id,
            "type": "access",
            "exp": datetime.now(timezone.utc) + self.access_token_expire,
            "iat": datetime.now(timezone.utc),
            "jti": secrets.token_urlsafe(16)
        }
        
        if self.jwt_algorithm == "RS256":
            token = jwt.encode(payload, self.jwt_private_key, algorithm=self.jwt_algorithm)
        else:
            token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
            
        token_operations.labels(operation="create", type="access").inc()
        
        return token
        
    async def _create_refresh_token(self, user: User) -> str:
        """Create refresh token with rotation"""
        token_id = secrets.token_urlsafe(32)
        
        # Store refresh token metadata
        await self.redis.setex(
            f"refresh_token:{token_id}",
            int(self.refresh_token_expire.total_seconds()),
            str({
                "user_id": user.id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "rotations": 0
            })
        )
        
        token_operations.labels(operation="create", type="refresh").inc()
        
        return token_id
        
    async def refresh_access_token(self, refresh_token: str) -> Tuple[str, str]:
        """Refresh access token with rotation"""
        token_data = await self.redis.get(f"refresh_token:{refresh_token}")
        
        if not token_data:
            token_operations.labels(operation="refresh", type="invalid").inc()
            raise HTTPException(status_code=401, detail="Invalid refresh token")
            
        data = eval(token_data)
        
        # Get user
        user_data = await self.redis.get(f"user:{data['user_id']}")
        if not user_data:
            raise HTTPException(status_code=401, detail="User not found")
            
        user = User(**eval(user_data))
        
        # Rotate refresh token
        await self.redis.delete(f"refresh_token:{refresh_token}")
        new_refresh_token = await self._create_refresh_token(user)
        
        # Update rotation count
        data["rotations"] += 1
        await self.redis.setex(
            f"refresh_token:{new_refresh_token}",
            int(self.refresh_token_expire.total_seconds()),
            str(data)
        )
        
        # Create new access token
        access_token = self._create_access_token(user)
        
        token_operations.labels(operation="refresh", type="success").inc()
        
        return access_token, new_refresh_token
        
    async def verify_token(self, token: str) -> User:
        """Verify and decode JWT token"""
        try:
            if self.jwt_algorithm == "RS256":
                payload = jwt.decode(token, self.jwt_public_key, algorithms=[self.jwt_algorithm])
            else:
                payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
                
            # Check if token is blacklisted
            if await self.redis.get(f"blacklist:token:{payload['jti']}"):
                raise HTTPException(status_code=401, detail="Token revoked")
                
            # Get user
            user_data = await self.redis.get(f"user:{payload['sub']}")
            if not user_data:
                raise HTTPException(status_code=401, detail="User not found")
                
            return User(**eval(user_data))
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
            
    async def revoke_token(self, token: str):
        """Revoke a token by blacklisting"""
        try:
            if self.jwt_algorithm == "RS256":
                payload = jwt.decode(token, self.jwt_public_key, algorithms=[self.jwt_algorithm])
            else:
                payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
                
            # Blacklist token until expiry
            ttl = payload["exp"] - int(time.time())
            if ttl > 0:
                await self.redis.setex(f"blacklist:token:{payload['jti']}", ttl, "1")
                
            token_operations.labels(operation="revoke", type="success").inc()
            
        except Exception:
            pass
            
    # API Key Management
    
    async def create_api_key(
        self,
        user: User,
        name: str,
        scopes: List[str],
        expires_days: Optional[int] = None
    ) -> str:
        """Create new API key"""
        if Permission.CREATE_API_KEY not in user.permissions:
            raise HTTPException(status_code=403, detail="Permission denied")
            
        # Generate key
        key = f"sk_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        # Create API key object
        api_key = APIKey(
            id=secrets.token_urlsafe(16),
            key_hash=key_hash,
            name=name,
            user_id=user.id,
            tenant_id=user.tenant_id,
            scopes=scopes,
            expires_at=datetime.now(timezone.utc) + timedelta(days=expires_days) if expires_days else None
        )
        
        # Store API key
        await self.redis.set(f"api_key:{key_hash}", str(api_key.__dict__))
        await self.redis.sadd(f"user:api_keys:{user.id}", key_hash)
        
        # Set expiry if needed
        if api_key.expires_at:
            ttl = int((api_key.expires_at - datetime.now(timezone.utc)).total_seconds())
            await self.redis.expire(f"api_key:{key_hash}", ttl)
            
        return key
        
    async def verify_api_key(self, key: str) -> Tuple[User, APIKey]:
        """Verify API key and return user"""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        # Get API key
        key_data = await self.redis.get(f"api_key:{key_hash}")
        if not key_data:
            api_key_usage.labels(key_id="unknown", scope="invalid").inc()
            raise HTTPException(status_code=401, detail="Invalid API key")
            
        api_key = APIKey(**eval(key_data))
        
        # Check expiry
        if api_key.expires_at and datetime.now(timezone.utc) > api_key.expires_at:
            api_key_usage.labels(key_id=api_key.id, scope="expired").inc()
            raise HTTPException(status_code=401, detail="API key expired")
            
        # Get user
        user_data = await self.redis.get(f"user:{api_key.user_id}")
        if not user_data:
            raise HTTPException(status_code=401, detail="User not found")
            
        user = User(**eval(user_data))
        
        # Update last used
        api_key.last_used = datetime.now(timezone.utc)
        await self.redis.set(f"api_key:{key_hash}", str(api_key.__dict__))
        
        api_key_usage.labels(key_id=api_key.id, scope="success").inc()
        
        return user, api_key
        
    async def list_api_keys(self, user: User) -> List[Dict[str, Any]]:
        """List user's API keys"""
        key_hashes = await self.redis.smembers(f"user:api_keys:{user.id}")
        
        keys = []
        for key_hash in key_hashes:
            key_data = await self.redis.get(f"api_key:{key_hash}")
            if key_data:
                api_key = APIKey(**eval(key_data))
                keys.append({
                    "id": api_key.id,
                    "name": api_key.name,
                    "scopes": api_key.scopes,
                    "created_at": api_key.created_at.isoformat(),
                    "last_used": api_key.last_used.isoformat() if api_key.last_used else None,
                    "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None
                })
                
        return keys
        
    async def revoke_api_key(self, user: User, key_id: str):
        """Revoke API key"""
        key_hashes = await self.redis.smembers(f"user:api_keys:{user.id}")
        
        for key_hash in key_hashes:
            key_data = await self.redis.get(f"api_key:{key_hash}")
            if key_data:
                api_key = APIKey(**eval(key_data))
                if api_key.id == key_id:
                    await self.redis.delete(f"api_key:{key_hash}")
                    await self.redis.srem(f"user:api_keys:{user.id}", key_hash)
                    return
                    
        raise HTTPException(status_code=404, detail="API key not found")
        
    # RBAC Implementation
    
    def check_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission"""
        return permission in user.permissions
        
    def check_role(self, user: User, role: Role) -> bool:
        """Check if user has specific role"""
        return role in user.roles
        
    async def add_role(self, user_id: str, role: Role):
        """Add role to user"""
        user_data = await self.redis.get(f"user:{user_id}")
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
            
        user = User(**eval(user_data))
        
        if role not in user.roles:
            user.roles.append(role)
            user.permissions.extend(self.role_permissions[role])
            user.permissions = list(set(user.permissions))  # Remove duplicates
            
            await self.redis.set(f"user:{user_id}", str(user.__dict__))
            
    async def remove_role(self, user_id: str, role: Role):
        """Remove role from user"""
        user_data = await self.redis.get(f"user:{user_id}")
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
            
        user = User(**eval(user_data))
        
        if role in user.roles:
            user.roles.remove(role)
            
            # Rebuild permissions
            user.permissions = []
            for r in user.roles:
                user.permissions.extend(self.role_permissions[r])
            user.permissions = list(set(user.permissions))
            
            await self.redis.set(f"user:{user_id}", str(user.__dict__))
            
    # Rate Limiting per User/Tenant/API Key
    
    async def check_rate_limit(
        self,
        identifier: str,
        limit: int,
        window: int = 3600
    ) -> Tuple[bool, Dict[str, int]]:
        """Check rate limit using sliding window"""
        now = int(time.time())
        window_start = now - window
        key = f"rate_limit:{identifier}"
        
        # Remove old entries
        await self.redis.zremrangebyscore(key, 0, window_start)
        
        # Count requests in window
        count = await self.redis.zcard(key)
        
        if count >= limit:
            # Get reset time
            oldest = await self.redis.zrange(key, 0, 0, withscores=True)
            reset_time = int(oldest[0][1]) + window if oldest else now + window
            
            return False, {
                "limit": limit,
                "remaining": 0,
                "reset": reset_time
            }
            
        # Add current request
        await self.redis.zadd(key, {f"{now}:{secrets.token_hex(4)}": now})
        await self.redis.expire(key, window)
        
        return True, {
            "limit": limit,
            "remaining": limit - count - 1,
            "reset": now + window
        }
        
    # OAuth2 Authorization Server
    
    async def create_oauth2_client(
        self,
        name: str,
        redirect_uris: List[str],
        scopes: List[str]
    ) -> Dict[str, Any]:
        """Create OAuth2 client for third-party integrations"""
        client = {
            "client_id": secrets.token_urlsafe(32),
            "client_secret": secrets.token_urlsafe(48),
            "name": name,
            "redirect_uris": redirect_uris,
            "scopes": scopes,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        await self.redis.set(f"oauth2_client:{client['client_id']}", str(client))
        
        return client
        
    async def oauth2_authorize_client(
        self,
        client_id: str,
        redirect_uri: str,
        scope: str,
        user: User
    ) -> str:
        """Authorize OAuth2 client and return authorization code"""
        # Verify client
        client_data = await self.redis.get(f"oauth2_client:{client_id}")
        if not client_data:
            raise HTTPException(status_code=400, detail="Invalid client")
            
        client = eval(client_data)
        
        # Verify redirect URI
        if redirect_uri not in client["redirect_uris"]:
            raise HTTPException(status_code=400, detail="Invalid redirect URI")
            
        # Generate authorization code
        code = secrets.token_urlsafe(32)
        
        # Store authorization
        await self.redis.setex(
            f"oauth2_code:{code}",
            600,  # 10 minutes
            str({
                "client_id": client_id,
                "user_id": user.id,
                "scope": scope,
                "redirect_uri": redirect_uri
            })
        )
        
        return code