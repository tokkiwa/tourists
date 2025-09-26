from supabase import create_client
import sys
import pathlib

# 現在のディレクトリをsys.pathに追加
current_dir = pathlib.Path(__file__).parent
sys.path.insert(0, str(current_dir))

from config import Config

SUPABASE_URL = Config.SUPABASE_URL
SUPABASE_KEY = Config.SUPABASE_KEY

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in the environment/config before creating the Supabase client.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)