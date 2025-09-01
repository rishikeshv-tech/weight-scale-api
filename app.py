from flask import Flask, jsonify, request, abort
from datetime import datetime, timedelta
import requests
import jwt
import hashlib
import hmac
import json
from functools import wraps
import logging
from typing import Dict, List, Optional
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeightScaleVendor:
    """Base class for weight scale vendor integrations"""
    
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        
    def authenticate(self) -> Optional[str]:
        """Authenticate with vendor API and return access token"""
        raise NotImplementedError
        
    def get_weight_data(self, user_id: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Fetch weight data for a user within date range"""
        raise NotImplementedError

class OmronVendor(WeightScaleVendor):
    """Omron weight scale vendor integration"""
    
    def __init__(self, api_key: str, api_secret: str, demo_mode: bool = False):
        # Use mock service for demo, real API for production
        if demo_mode or api_key == 'demo':
            base_url = "http://localhost:5001"
        else:
            base_url = "https://api.omronhealthcare.com"
        super().__init__(api_key, api_secret, base_url)
        self.access_token = None
        self.demo_mode = demo_mode or api_key == 'demo'
        
    def authenticate(self) -> Optional[str]:
        """Authenticate with Omron API"""
        try:
            auth_url = f"{self.base_url}/oauth/token"
            headers = {
                'Content-Type': 'application/json',
            }
            
            if self.demo_mode:
                # Simple mock authentication
                payload = {'grant_type': 'client_credentials'}
            else:
                # Real Omron authentication
                headers['Authorization'] = f'Basic {self.api_key}:{self.api_secret}'
                payload = {
                    'grant_type': 'client_credentials',
                    'scope': 'weight_data'
                }
            
            response = requests.post(auth_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                self.access_token = data.get('access_token')
                logger.info(f"Omron authentication successful (demo_mode: {self.demo_mode})")
                return self.access_token
            else:
                logger.error(f"Omron authentication failed: {response.status_code}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Omron authentication error: {str(e)}")
            return None
    
    def get_weight_data(self, user_id: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Fetch weight data from Omron API"""
        if not self.access_token:
            if not self.authenticate():
                return []
        
        try:
            data_url = f"{self.base_url}/v1/users/{user_id}/weight"
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }
            
            params = {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            }
            
            response = requests.get(data_url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                measurements = data.get('measurements', [])
                logger.info(f"Retrieved {len(measurements)} measurements for user {user_id}")
                return self._normalize_weight_data(measurements)
            elif response.status_code == 401:
                # Token expired, re-authenticate
                logger.info("Token expired, re-authenticating...")
                if self.authenticate():
                    headers['Authorization'] = f'Bearer {self.access_token}'
                    response = requests.get(data_url, headers=headers, params=params, timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        measurements = data.get('measurements', [])
                        return self._normalize_weight_data(measurements)
            
            logger.error(f"Failed to fetch Omron data: {response.status_code} - {response.text}")
            return []
            
        except requests.RequestException as e:
            logger.error(f"Omron data fetch error: {str(e)}")
            return []
    
    def _normalize_weight_data(self, raw_data: List[Dict]) -> List[Dict]:
        """Normalize Omron data to standard format"""
        normalized = []
        for measurement in raw_data:
            normalized.append({
                'timestamp': measurement.get('measured_at'),
                'weight_kg': measurement.get('weight'),
                'bmi': measurement.get('bmi'),
                'body_fat_percentage': measurement.get('body_fat'),
                'muscle_mass_kg': measurement.get('muscle_mass'),
                'bone_mass_kg': measurement.get('bone_mass'),
                'metabolic_age': measurement.get('metabolic_age'),
                'device_id': measurement.get('device_id'),
                'vendor': 'omron'
            })
        return normalized

class WithingsVendor(WeightScaleVendor):
    """Withings weight scale vendor integration"""
    
    def __init__(self, api_key: str, api_secret: str, demo_mode: bool = False):
        if demo_mode or api_key == 'demo':
            base_url = "http://localhost:5001"  # Could use same mock service
        else:
            base_url = "https://wbsapi.withings.net"
        super().__init__(api_key, api_secret, base_url)
        self.access_token = None
        self.demo_mode = demo_mode or api_key == 'demo'
    
    def authenticate(self) -> Optional[str]:
        """Authenticate with Withings API"""
        # For demo purposes, return a mock token
        if self.demo_mode:
            self.access_token = "withings_mock_token"
            return self.access_token
        
        # Real Withings authentication would go here
        logger.warning("Withings integration not fully implemented yet")
        return None
    
    def get_weight_data(self, user_id: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Fetch weight data from Withings API"""
        if self.demo_mode:
            # Return some mock data for demo
            return [{
                'timestamp': datetime.now().isoformat(),
                'weight_kg': 68.5,
                'bmi': 22.1,
                'body_fat_percentage': 16.8,
                'muscle_mass_kg': 32.1,
                'bone_mass_kg': 2.9,
                'metabolic_age': 25,
                'device_id': 'WITHINGS_SCALE_001',
                'vendor': 'withings'
            }]
        
        # Real implementation would go here
        logger.warning("Withings data fetch not implemented yet")
        return []

# Vendor factory
VENDORS = {
    'omron': OmronVendor,
    'withings': WithingsVendor
}

def get_vendor(vendor_name: str, api_key: str, api_secret: str) -> Optional[WeightScaleVendor]:
    """Factory function to get vendor instance"""
    vendor_class = VENDORS.get(vendor_name.lower())
    if vendor_class:
        return vendor_class(api_key, api_secret, demo_mode=(api_key == 'demo'))
    return None

# Authentication decorator
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            abort(401, description="API key required. Include 'X-API-Key' header.")
        
        # Validate API key
        if not validate_api_key(api_key):
            abort(401, description="Invalid API key")
        
        return f(*args, **kwargs)
    return decorated_function

def validate_api_key(api_key: str) -> bool:
    """Validate API key - implement your own logic"""
    # Get valid keys from environment variable
    valid_keys = os.environ.get('VALID_API_KEYS', 'demo-key-123').split(',')
    valid_keys = [key.strip() for key in valid_keys if key.strip()]
    
    logger.info(f"Validating API key against {len(valid_keys)} valid keys")
    return api_key in valid_keys

# Error handling helper
def create_error_response(message: str, status_code: int = 400, details: str = None):
    """Create consistent error responses"""
    response = {
        'error': message,
        'timestamp': datetime.utcnow().isoformat(),
        'status_code': status_code
    }
    if details:
        response['details'] = details
    return jsonify(response), status_code

# CORS helper for browser testing
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-API-Key')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# API Routes

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        'name': 'Weight Scale Data API',
        'version': '1.0.0',
        'description': 'Unified API for cellular weight scale data aggregation',
        'endpoints': {
            'health': '/api/v1/health',
            'vendors': '/api/v1/vendors',
            'weight_data': '/api/v1/weight/<vendor>/<user_id>',
            'latest_weight': '/api/v1/weight/<vendor>/<user_id>/latest',
            'weight_summary': '/api/v1/weight/<vendor>/<user_id>/summary'
        },
        'authentication': 'API key required in X-API-Key header',
        'demo_credentials': {
            'api_key': 'demo-key-123',
            'vendor_api_key': 'demo',
            'vendor_api_secret': 'demo',
            'test_user_id': 'demo-user-001'
        }
    })

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0',
        'environment': 'demo' if 'demo' in os.environ.get('VALID_API_KEYS', '') else 'production'
    })

@app.route('/api/v1/vendors', methods=['GET'])
@require_api_key
def list_vendors():
    """List available weight scale vendors"""
    return jsonify({
        'vendors': [
            {
                'name': 'omron',
                'display_name': 'Omron HealthCare',
                'description': 'Omron cellular weight scales and body composition monitors',
                'status': 'active'
            },
            {
                'name': 'withings',
                'display_name': 'Withings (Nokia Health)',
                'description': 'Withings smart scales and health devices',
                'status': 'demo_only'
            }
        ],
        'count': len(VENDORS),
        'demo_note': 'Use vendor_api_key=demo for testing'
    })

@app.route('/api/v1/weight/<vendor>/<user_id>', methods=['GET'])
@require_api_key
def get_weight_data(vendor: str, user_id: str):
    """
    Fetch weight data for a user from specified vendor
    
    Query parameters:
    - start_date: ISO format date (optional, defaults to 30 days ago)
    - end_date: ISO format date (optional, defaults to now)
    - vendor_api_key: Vendor API key (required)
    - vendor_api_secret: Vendor API secret (required)
    """
    try:
        # Get query parameters
        vendor_api_key = request.args.get('vendor_api_key')
        vendor_api_secret = request.args.get('vendor_api_secret')
        
        if not vendor_api_key or not vendor_api_secret:
            return create_error_response(
                'Vendor API credentials required',
                400,
                'Include vendor_api_key and vendor_api_secret query parameters'
            )
        
        # Parse date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)
        
        if request.args.get('start_date'):
            try:
                start_date_str = request.args.get('start_date')
                start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
            except ValueError:
                return create_error_response(
                    'Invalid start_date format',
                    400,
                    'Use ISO format: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD'
                )
        
        if request.args.get('end_date'):
            try:
                end_date_str = request.args.get('end_date')
                end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
            except ValueError:
                return create_error_response(
                    'Invalid end_date format',
                    400,
                    'Use ISO format: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD'
                )
        
        # Validate date range
        if start_date > end_date:
            return create_error_response('start_date must be before end_date', 400)
        
        # Check if date range is too large (prevent abuse)
        if (end_date - start_date).days > 365:
            return create_error_response(
                'Date range too large', 
                400, 
                'Maximum range is 365 days'
            )
        
        # Get vendor instance
        vendor_instance = get_vendor(vendor, vendor_api_key, vendor_api_secret)
        if not vendor_instance:
            return create_error_response(
                f'Unsupported vendor: {vendor}',
                400,
                f'Supported vendors: {", ".join(VENDORS.keys())}'
            )
        
        # Fetch weight data
        logger.info(f"Fetching weight data for {vendor}/{user_id} from {start_date} to {end_date}")
        weight_data = vendor_instance.get_weight_data(user_id, start_date, end_date)
        
        return jsonify({
            'user_id': user_id,
            'vendor': vendor,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'measurements': weight_data,
            'count': len(weight_data),
            'query_timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error fetching weight data: {str(e)}")
        return create_error_response('Internal server error', 500)

@app.route('/api/v1/weight/<vendor>/<user_id>/latest', methods=['GET'])
@require_api_key
def get_latest_weight(vendor: str, user_id: str):
    """Get the most recent weight measurement for a user"""
    try:
        vendor_api_key = request.args.get('vendor_api_key')
        vendor_api_secret = request.args.get('vendor_api_secret')
        
        if not vendor_api_key or not vendor_api_secret:
            return create_error_response('Vendor API credentials required', 400)
        
        # Get last 7 days of data to find most recent
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)
        
        vendor_instance = get_vendor(vendor, vendor_api_key, vendor_api_secret)
        if not vendor_instance:
            return create_error_response(f'Unsupported vendor: {vendor}', 400)
        
        weight_data = vendor_instance.get_weight_data(user_id, start_date, end_date)
        
        if not weight_data:
            return create_error_response('No recent weight data found', 404)
        
        # Sort by timestamp and get latest
        try:
            latest = max(weight_data, key=lambda x: datetime.fromisoformat(x['timestamp'].replace('Z', '+00:00')))
        except (ValueError, KeyError):
            # Fallback: just take the last item
            latest = weight_data[-1]
        
        return jsonify({
            'user_id': user_id,
            'vendor': vendor,
            'latest_measurement': latest,
            'query_timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error fetching latest weight: {str(e)}")
        return create_error_response('Internal server error', 500)

@app.route('/api/v1/weight/<vendor>/<user_id>/summary', methods=['GET'])
@require_api_key
def get_weight_summary(vendor: str, user_id: str):
    """Get weight summary statistics for a user"""
    try:
        vendor_api_key = request.args.get('vendor_api_key')
        vendor_api_secret = request.args.get('vendor_api_secret')
        
        if not vendor_api_key or not vendor_api_secret:
            return create_error_response('Vendor API credentials required', 400)
        
        # Get configurable period (default 90 days)
        period_days = int(request.args.get('period_days', 90))
        period_days = min(period_days, 365)  # Max 1 year
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)
        
        vendor_instance = get_vendor(vendor, vendor_api_key, vendor_api_secret)
        if not vendor_instance:
            return create_error_response(f'Unsupported vendor: {vendor}', 400)
        
        weight_data = vendor_instance.get_weight_data(user_id, start_date, end_date)
        
        if not weight_data:
            return create_error_response('No weight data found for the specified period', 404)
        
        # Calculate statistics
        weights = [m['weight_kg'] for m in weight_data if m.get('weight_kg') is not None]
        
        if not weights:
            return create_error_response('No valid weight measurements found', 404)
        
        # Sort data by timestamp for trend analysis
        try:
            sorted_data = sorted(weight_data, key=lambda x: datetime.fromisoformat(x['timestamp'].replace('Z', '+00:00')))
        except (ValueError, KeyError):
            sorted_data = weight_data
        
        sorted_weights = [m['weight_kg'] for m in sorted_data if m.get('weight_kg') is not None]
        
        summary = {
            'user_id': user_id,
            'vendor': vendor,
            'period_days': period_days,
            'measurement_count': len(weights),
            'weight_stats': {
                'current_kg': sorted_weights[-1] if sorted_weights else None,
                'min_kg': round(min(weights), 1),
                'max_kg': round(max(weights), 1),
                'avg_kg': round(sum(weights) / len(weights), 1),
                'weight_change_kg': round(sorted_weights[-1] - sorted_weights[0], 1) if len(sorted_weights) > 1 else 0,
                'weight_trend': 'increasing' if len(sorted_weights) > 1 and sorted_weights[-1] > sorted_weights[0] else 'decreasing' if len(sorted_weights) > 1 else 'stable'
            },
            'first_measurement': sorted_data[0]['timestamp'] if sorted_data else None,
            'last_measurement': sorted_data[-1]['timestamp'] if sorted_data else None,
            'query_timestamp': datetime.utcnow().isoformat()
        }
        
        # Add BMI stats if available
        bmis = [m['bmi'] for m in weight_data if m.get('bmi') is not None]
        if bmis:
            summary['bmi_stats'] = {
                'current': round(bmis[-1], 1) if bmis else None,
                'avg': round(sum(bmis) / len(bmis), 1),
                'min': round(min(bmis), 1),
                'max': round(max(bmis), 1)
            }
        
        return jsonify(summary)
        
    except Exception as e:
        logger.error(f"Error calculating weight summary: {str(e)}")
        return create_error_response('Internal server error', 500)

# Error handlers
@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        'error': error.description,
        'status_code': 400,
        'timestamp': datetime.utcnow().isoformat()
    }), 400

@app.errorhandler(401)
def unauthorized(error):
    return jsonify({
        'error': error.description,
        'status_code': 401,
        'timestamp': datetime.utcnow().isoformat(),
        'help': 'Include X-API-Key header with valid API key'
    }), 401

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Resource not found',
        'status_code': 404,
        'timestamp': datetime.utcnow().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'status_code': 500,
        'timestamp': datetime.utcnow().isoformat()
    }), 500

# Health and info routes
@app.route('/api/v1/info', methods=['GET'])
def api_info():
    """API information and usage guide"""
    return jsonify({
        'api_name': 'Weight Scale Data API',
        'version': '1.0.0',
        'description': 'Unified API for aggregating cellular weight scale data from multiple vendors',
        'supported_vendors': list(VENDORS.keys()),
        'authentication': {
            'method': 'API Key',
            'header': 'X-API-Key',
            'demo_key': 'demo-key-123'
        },
        'demo_usage': {
            'vendor_credentials': {
                'vendor_api_key': 'demo',
                'vendor_api_secret': 'demo'
            },
            'test_user': 'demo-user-001',
            'example_request': '/api/v1/weight/omron/demo-user-001?vendor_api_key=demo&vendor_api_secret=demo'
        },
        'endpoints': {
            'GET /': 'API overview',
            'GET /api/v1/health': 'Health check',
            'GET /api/v1/vendors': 'List supported vendors',
            'GET /api/v1/weight/{vendor}/{user_id}': 'Get weight data',
            'GET /api/v1/weight/{vendor}/{user_id}/latest': 'Get latest measurement',
            'GET /api/v1/weight/{vendor}/{user_id}/summary': 'Get weight summary statistics'
        }
    })

if __name__ == '__main__':
    # Set default environment variables for development
    if not os.environ.get('SECRET_KEY'):
        os.environ['SECRET_KEY'] = 'dev-secret-key-change-in-production'
    
    if not os.environ.get('VALID_API_KEYS'):
        os.environ['VALID_API_KEYS'] = 'demo-key-123,dev-key-456'
    
    logger.info("Starting Weight Scale API...")
    logger.info(f"Valid API keys configured: {len(os.environ.get('VALID_API_KEYS', '').split(','))}")
    logger.info("Demo mode available with API key: demo-key-123")
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)
