#!/usr/bin/env python3
"""
Setup script for AI Chatbot Enterprise Systems
"""

from setuptools import setup, find_packages

setup(
    name="ai-chatbot-enterprise",
    version="1.0.0",
    description="Enterprise AI Chatbot System with advanced authentication, observability, and FinOps",
    author="Christopher Bratkovics",
    author_email="chris@enterprise.com",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "redis>=5.0.1",
        "openai>=1.6.1",
        "anthropic>=0.8.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
        ],
        "finops": [
            "streamlit>=1.28.0",
            "pandas>=2.1.0",
            "plotly>=5.17.0",
            "boto3>=1.34.0",
        ],
        "monitoring": [
            "prometheus-client>=0.19.0",
            "grafana-api>=1.0.3",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "finops-dashboard=finops.dashboard.finops_dashboard:main",
            "dr-automation=infrastructure.disaster_recovery.dr_automation:main",
            "auth-service=api.app.auth.unified_auth_service:main",
        ],
    },
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md"],
    },
    include_package_data=True,
    zip_safe=False,
)