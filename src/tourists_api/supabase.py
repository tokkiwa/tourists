from supabase import create_client

from .config import Config

SUPABASE_URL = Config.SUPABASE_URL
SUPABASE_KEY = Config.SUPABASE_KEY

def get_supabase_client():
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in the environment/config before creating the Supabase client.")
    return create_client(SUPABASE_URL, SUPABASE_KEY)