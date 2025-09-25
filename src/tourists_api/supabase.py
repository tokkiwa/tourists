from supabase import create_client

from .config import Config

SUPABASE_URL = Config.SUPABASE_URL
SUPABASE_KEY = Config.SUPABASE_KEY
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)