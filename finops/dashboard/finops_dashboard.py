#!/usr/bin/env python3
"""
FinOps Interactive Dashboard
Real-time cost monitoring, analytics, and management interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import boto3
import redis
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

# Dashboard configuration
st.set_page_config(
    page_title="FinOps Cost Management Platform",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

class FinOpsDashboard:
    """Interactive FinOps dashboard with real-time cost analytics"""
    
    def __init__(self):
        self.setup_connections()
        self.load_sample_data()
    
    def setup_connections(self):
        """Initialize database and cache connections"""
        # These would be replaced with actual connections in production
        self.redis_client = None  # redis.Redis(host='localhost', port=6379, db=0)
        self.db_engine = None     # create_engine('postgresql://...')
        self.aws_session = None   # boto3.Session()
    
    def load_sample_data(self):
        """Load sample data for demonstration"""
        # Cost data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        
        services = ['EC2', 'RDS', 'S3', 'Lambda', 'API Gateway', 'CloudWatch', 'AI APIs']
        teams = ['Backend', 'Frontend', 'Data', 'DevOps']
        
        # Generate sample cost data
        np.random.seed(42)
        cost_data = []
        
        for date in dates:
            for service in services:
                for team in teams:
                    base_cost = {
                        'EC2': 1000,
                        'RDS': 500,
                        'S3': 200,
                        'Lambda': 100,
                        'API Gateway': 50,
                        'CloudWatch': 75,
                        'AI APIs': 800
                    }[service]
                    
                    # Add some seasonal variation and team-specific patterns
                    seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * date.dayofyear / 365)
                    team_factor = {
                        'Backend': 1.0,
                        'Frontend': 0.6,
                        'Data': 1.4,
                        'DevOps': 0.8
                    }[team]
                    
                    # Add random variation
                    daily_cost = base_cost * seasonal_factor * team_factor * (0.8 + 0.4 * np.random.random())
                    
                    cost_data.append({
                        'date': date,
                        'service': service,
                        'team': team,
                        'cost': daily_cost,
                        'region': np.random.choice(['us-east-1', 'us-west-2', 'eu-west-1']),
                        'environment': np.random.choice(['production', 'development', 'staging'], p=[0.7, 0.2, 0.1])
                    })
        
        self.cost_df = pd.DataFrame(cost_data)
        
        # Anomaly data
        self.anomalies_df = pd.DataFrame([
            {
                'date': '2024-11-25',
                'service': 'AI APIs',
                'team': 'Data',
                'severity': 'Critical',
                'current_cost': 5200,
                'expected_cost': 2800,
                'variance_percent': 85.7,
                'description': 'Unusual spike in AI API usage'
            },
            {
                'date': '2024-11-20',
                'service': 'EC2',
                'team': 'Backend',
                'severity': 'High',
                'current_cost': 1800,
                'expected_cost': 1200,
                'variance_percent': 50.0,
                'description': 'Higher than expected compute usage'
            },
            {
                'date': '2024-11-18',
                'service': 'RDS',
                'team': 'Backend',
                'severity': 'Medium',
                'current_cost': 720,
                'expected_cost': 500,
                'variance_percent': 44.0,
                'description': 'Database costs above normal range'
            }
        ])
        
        # Savings recommendations
        self.recommendations_df = pd.DataFrame([
            {
                'category': 'Right-sizing',
                'service': 'EC2',
                'team': 'Backend',
                'description': 'Downsize over-provisioned instances',
                'monthly_savings': 2400,
                'implementation_effort': 'Low',
                'confidence': 0.85
            },
            {
                'category': 'Reserved Instances',
                'service': 'RDS',
                'team': 'Backend',
                'description': 'Purchase 3-year RDS reserved instances',
                'monthly_savings': 1800,
                'implementation_effort': 'Medium',
                'confidence': 0.90
            },
            {
                'category': 'AI API Optimization',
                'service': 'AI APIs',
                'team': 'Data',
                'description': 'Optimize prompts to reduce token usage',
                'monthly_savings': 1600,
                'implementation_effort': 'Medium',
                'confidence': 0.75
            },
            {
                'category': 'Storage Optimization',
                'service': 'S3',
                'team': 'All',
                'description': 'Implement S3 lifecycle policies',
                'monthly_savings': 800,
                'implementation_effort': 'Low',
                'confidence': 0.70
            },
            {
                'category': 'Reserved Instances',
                'service': 'Lambda',
                'team': 'Backend',
                'description': 'Use provisioned concurrency for stable workloads',
                'monthly_savings': 600,
                'implementation_effort': 'High',
                'confidence': 0.65
            }
        ])
    
    def render_header(self):
        """Render dashboard header"""
        st.title("💰 FinOps Cost Management Platform")
        st.markdown("---")
        
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Calculate current month metrics
        current_month = datetime.now().replace(day=1)
        month_data = self.cost_df[self.cost_df['date'] >= current_month]
        total_monthly_cost = month_data['cost'].sum()
        
        with col1:
            st.metric(
                label="Monthly Spend",
                value=f"${total_monthly_cost:,.0f}",
                delta=f"{((total_monthly_cost / 180000) - 1) * 100:.1f}%"
            )
        
        with col2:
            st.metric(
                label="Active Anomalies",
                value=len(self.anomalies_df),
                delta=f"-{np.random.randint(1, 4)}"
            )
        
        with col3:
            potential_savings = self.recommendations_df['monthly_savings'].sum()
            st.metric(
                label="Potential Savings",
                value=f"${potential_savings:,.0f}",
                delta=f"{np.random.randint(200, 800)}"
            )
        
        with col4:
            st.metric(
                label="Cost Efficiency",
                value="87%",
                delta="3.2%"
            )
        
        with col5:
            st.metric(
                label="Budget Utilization",
                value="73%",
                delta="-5.1%"
            )
    
    def render_cost_trends(self):
        """Render cost trends visualization"""
        st.subheader("📈 Cost Trends")
        
        # Time period selector
        col1, col2 = st.columns([3, 1])
        
        with col2:
            time_period = st.selectbox(
                "Time Period",
                ["Last 30 Days", "Last 90 Days", "Last 6 Months", "Last Year"],
                index=2
            )
            
            group_by = st.selectbox(
                "Group By",
                ["Service", "Team", "Environment", "Region"]
            )
        
        # Filter data based on selection
        days_map = {
            "Last 30 Days": 30,
            "Last 90 Days": 90,
            "Last 6 Months": 180,
            "Last Year": 365
        }
        
        cutoff_date = datetime.now() - timedelta(days=days_map[time_period])
        filtered_df = self.cost_df[self.cost_df['date'] >= cutoff_date].copy()
        
        # Aggregate data
        group_col = group_by.lower()
        if group_col == 'service':
            daily_costs = filtered_df.groupby(['date', 'service'])['cost'].sum().reset_index()
            
            fig = px.line(
                daily_costs,
                x='date',
                y='cost',
                color='service',
                title=f"Daily Costs by Service - {time_period}",
                labels={'cost': 'Cost ($)', 'date': 'Date'}
            )
        else:
            daily_costs = filtered_df.groupby(['date', group_col])['cost'].sum().reset_index()
            
            fig = px.line(
                daily_costs,
                x='date',
                y='cost',
                color=group_col,
                title=f"Daily Costs by {group_by} - {time_period}",
                labels={'cost': 'Cost ($)', 'date': 'Date'}
            )
        
        fig.update_layout(
            height=400,
            xaxis_title="Date",
            yaxis_title="Cost ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_service_breakdown(self):
        """Render service cost breakdown"""
        st.subheader("🔧 Service Cost Breakdown")
        
        col1, col2 = st.columns(2)
        
        # Current month data
        current_month = datetime.now().replace(day=1)
        month_data = self.cost_df[self.cost_df['date'] >= current_month]
        
        with col1:
            # Service pie chart
            service_costs = month_data.groupby('service')['cost'].sum().reset_index()
            service_costs = service_costs.sort_values('cost', ascending=False)
            
            fig_pie = px.pie(
                service_costs,
                values='cost',
                names='service',
                title="Cost Distribution by Service (Current Month)"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Team breakdown
            team_costs = month_data.groupby('team')['cost'].sum().reset_index()
            team_costs = team_costs.sort_values('cost', ascending=True)
            
            fig_bar = px.bar(
                team_costs,
                x='cost',
                y='team',
                orientation='h',
                title="Cost by Team (Current Month)",
                labels={'cost': 'Cost ($)', 'team': 'Team'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    def render_anomaly_detection(self):
        """Render anomaly detection section"""
        st.subheader("🚨 Cost Anomaly Detection")
        
        # Anomaly severity distribution
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Anomaly timeline
            fig_timeline = go.Figure()
            
            severity_colors = {
                'Critical': '#ff4444',
                'High': '#ff8800',
                'Medium': '#ffaa00',
                'Low': '#44aa44'
            }
            
            for severity in self.anomalies_df['severity'].unique():
                severity_data = self.anomalies_df[self.anomalies_df['severity'] == severity]
                
                fig_timeline.add_trace(go.Scatter(
                    x=pd.to_datetime(severity_data['date']),
                    y=severity_data['variance_percent'],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=severity_colors[severity],
                        line=dict(width=2, color='white')
                    ),
                    name=severity,
                    text=severity_data['description'],
                    hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Variance: %{y:.1f}%<extra></extra>'
                ))
            
            fig_timeline.update_layout(
                title="Cost Anomalies Over Time",
                xaxis_title="Date",
                yaxis_title="Variance (%)",
                height=300
            )
            
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        with col2:
            st.write("**Recent Anomalies**")
            
            for _, anomaly in self.anomalies_df.iterrows():
                severity_color = {
                    'Critical': '🔴',
                    'High': '🟠',
                    'Medium': '🟡',
                    'Low': '🟢'
                }[anomaly['severity']]
                
                st.write(f"{severity_color} **{anomaly['service']}** ({anomaly['team']})")
                st.write(f"Variance: +{anomaly['variance_percent']:.1f}%")
                st.write(f"Impact: ${anomaly['current_cost'] - anomaly['expected_cost']:.0f}")
                st.write("---")
    
    def render_savings_recommendations(self):
        """Render savings recommendations section"""
        st.subheader("💡 Savings Recommendations")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Recommendations table
            recommendations_display = self.recommendations_df.copy()
            recommendations_display['monthly_savings'] = recommendations_display['monthly_savings'].apply(
                lambda x: f"${x:,.0f}"
            )
            recommendations_display['confidence'] = recommendations_display['confidence'].apply(
                lambda x: f"{x:.0%}"
            )
            
            st.dataframe(
                recommendations_display[['category', 'service', 'description', 'monthly_savings', 'implementation_effort', 'confidence']],
                column_config={
                    'category': 'Category',
                    'service': 'Service',
                    'description': 'Description',
                    'monthly_savings': 'Monthly Savings',
                    'implementation_effort': 'Effort',
                    'confidence': 'Confidence'
                },
                hide_index=True,
                use_container_width=True
            )
        
        with col2:
            # Savings potential by category
            category_savings = self.recommendations_df.groupby('category')['monthly_savings'].sum().reset_index()
            category_savings = category_savings.sort_values('monthly_savings', ascending=True)
            
            fig_savings = px.bar(
                category_savings,
                x='monthly_savings',
                y='category',
                orientation='h',
                title="Savings Potential by Category",
                labels={'monthly_savings': 'Monthly Savings ($)', 'category': 'Category'}
            )
            
            st.plotly_chart(fig_savings, use_container_width=True)
    
    def render_api_optimization(self):
        """Render AI API optimization section"""
        st.subheader("🤖 AI API Cost Optimization")
        
        # Sample API usage data
        api_data = pd.DataFrame([
            {'provider': 'OpenAI', 'model': 'GPT-4', 'requests': 15000, 'tokens': 2100000, 'cost': 4200},
            {'provider': 'OpenAI', 'model': 'GPT-3.5-Turbo', 'requests': 45000, 'tokens': 1800000, 'cost': 1800},
            {'provider': 'Anthropic', 'model': 'Claude-3-Opus', 'requests': 8000, 'tokens': 1600000, 'cost': 2400},
            {'provider': 'Anthropic', 'model': 'Claude-3-Sonnet', 'requests': 25000, 'tokens': 2000000, 'cost': 1200},
            {'provider': 'AWS Bedrock', 'model': 'Claude-v2', 'requests': 12000, 'tokens': 1400000, 'cost': 1400}
        ])
        
        api_data['cost_per_request'] = api_data['cost'] / api_data['requests']
        api_data['cost_per_1k_tokens'] = api_data['cost'] / (api_data['tokens'] / 1000)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Cost per request
            fig_cpr = px.bar(
                api_data,
                x='model',
                y='cost_per_request',
                color='provider',
                title="Cost per Request by Model",
                labels={'cost_per_request': 'Cost per Request ($)'}
            )
            fig_cpr.update_xaxis(tickangle=45)
            st.plotly_chart(fig_cpr, use_container_width=True)
        
        with col2:
            # Token efficiency
            fig_tokens = px.scatter(
                api_data,
                x='tokens',
                y='cost',
                size='requests',
                color='provider',
                hover_name='model',
                title="Token Usage vs Cost",
                labels={'tokens': 'Total Tokens', 'cost': 'Total Cost ($)'}
            )
            st.plotly_chart(fig_tokens, use_container_width=True)
        
        with col3:
            # Usage distribution
            fig_usage = px.pie(
                api_data,
                values='requests',
                names='model',
                title="Request Distribution by Model"
            )
            st.plotly_chart(fig_usage, use_container_width=True)
    
    def render_chargeback_reports(self):
        """Render chargeback reporting section"""
        st.subheader("📊 Chargeback Reports")
        
        # Month selector
        col1, col2 = st.columns([1, 3])
        
        with col1:
            selected_month = st.selectbox(
                "Select Month",
                ["November 2024", "October 2024", "September 2024", "August 2024"]
            )
            
            report_format = st.selectbox(
                "Report Format",
                ["Interactive", "PDF", "CSV", "JSON"]
            )
        
        with col2:
            # Team cost breakdown for selected month
            current_month = datetime.now().replace(day=1)
            month_data = self.cost_df[self.cost_df['date'] >= current_month]
            
            # Calculate team costs with overhead allocation
            team_costs = month_data.groupby(['team', 'service'])['cost'].sum().reset_index()
            team_totals = team_costs.groupby('team')['cost'].sum().reset_index()
            
            # Add overhead (10% for engineering teams)
            team_totals['overhead'] = team_totals['cost'] * 0.1
            team_totals['total_with_overhead'] = team_totals['cost'] + team_totals['overhead']
            
            # Create pivot table for heatmap
            cost_matrix = month_data.groupby(['team', 'service'])['cost'].sum().unstack(fill_value=0)
            
            fig_heatmap = px.imshow(
                cost_matrix.values,
                x=cost_matrix.columns,
                y=cost_matrix.index,
                aspect="auto",
                title="Team vs Service Cost Matrix",
                labels=dict(x="Service", y="Team", color="Cost ($)"),
                color_continuous_scale="Viridis"
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Detailed breakdown table
        st.write("**Detailed Cost Breakdown**")
        
        chargeback_summary = team_totals.copy()
        chargeback_summary['cost'] = chargeback_summary['cost'].apply(lambda x: f"${x:,.0f}")
        chargeback_summary['overhead'] = chargeback_summary['overhead'].apply(lambda x: f"${x:,.0f}")
        chargeback_summary['total_with_overhead'] = chargeback_summary['total_with_overhead'].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(
            chargeback_summary,
            column_config={
                'team': 'Team',
                'cost': 'Direct Costs',
                'overhead': 'Overhead (10%)',
                'total_with_overhead': 'Total Chargeback'
            },
            hide_index=True
        )
        
        # Download buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Generate PDF Report"):
                st.success("PDF report generated successfully!")
        
        with col2:
            if st.button("Export to CSV"):
                st.success("CSV export completed!")
        
        with col3:
            if st.button("Send Email Report"):
                st.success("Email report sent to stakeholders!")
    
    def render_sidebar(self):
        """Render sidebar with filters and controls"""
        st.sidebar.title("Dashboard Controls")
        
        # Time range filter
        st.sidebar.subheader("Time Range")
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            max_value=datetime.now()
        )
        
        # Team filter
        st.sidebar.subheader("Filters")
        selected_teams = st.sidebar.multiselect(
            "Teams",
            options=self.cost_df['team'].unique(),
            default=self.cost_df['team'].unique()
        )
        
        # Service filter
        selected_services = st.sidebar.multiselect(
            "Services",
            options=self.cost_df['service'].unique(),
            default=self.cost_df['service'].unique()
        )
        
        # Environment filter
        selected_environments = st.sidebar.multiselect(
            "Environments",
            options=self.cost_df['environment'].unique(),
            default=self.cost_df['environment'].unique()
        )
        
        # Refresh data button
        st.sidebar.subheader("Actions")
        if st.sidebar.button("Refresh Data"):
            st.experimental_rerun()
        
        # Export options
        st.sidebar.subheader("Export")
        if st.sidebar.button("Export Dashboard"):
            st.sidebar.success("Dashboard exported!")
        
        return {
            'date_range': date_range,
            'teams': selected_teams,
            'services': selected_services,
            'environments': selected_environments
        }
    
    def run(self):
        """Run the dashboard"""
        # Render sidebar
        filters = self.render_sidebar()
        
        # Main dashboard
        self.render_header()
        
        # Dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Cost Overview",
            "🚨 Anomalies",
            "💡 Recommendations",
            "🤖 AI Optimization",
            "📊 Chargeback"
        ])
        
        with tab1:
            self.render_cost_trends()
            st.markdown("---")
            self.render_service_breakdown()
        
        with tab2:
            self.render_anomaly_detection()
        
        with tab3:
            self.render_savings_recommendations()
        
        with tab4:
            self.render_api_optimization()
        
        with tab5:
            self.render_chargeback_reports()

# Main execution
if __name__ == "__main__":
    dashboard = FinOpsDashboard()
    dashboard.run()