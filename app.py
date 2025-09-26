#!/usr/bin/env python3
"""
Main application entry point for the Tourists API
"""
import os
from src.tourists_api import create_app

app = create_app()

if __name__ == '__main__':
    # 開発環境での起動設定
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"🚀 Starting Tourists API server on port {port}")
    print(f"🌍 Available at: http://localhost:{port}")
    print(f"📚 API endpoints available at: http://localhost:{port}/api/")
    print(f"🔍 Debug mode: {debug}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
