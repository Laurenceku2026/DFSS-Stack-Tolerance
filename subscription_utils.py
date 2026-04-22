# subscription_utils.py (用于工具 App)
import streamlit as st
from supabase import create_client

@st.cache_resource
def get_supabase_admin_client():
    """管理员客户端（service_role key，绕过 RLS）"""
    try:
        url = st.secrets["connections"]["supabase"]["SUPABASE_URL"]
        key = st.secrets["connections"]["supabase"]["SUPABASE_SERVICE_ROLE_KEY"]
        return create_client(url, key)
    except Exception as e:
        st.error(f"管理员客户端初始化失败: {e}")
        return None

def get_user_subscription(email: str):
    """使用管理员客户端获取订阅信息（绕过 RLS）"""
    supabase = get_supabase_admin_client()
    if supabase is None:
        return {"subscription_status": "free", "usage_count": 0, "usage_limit": 10}
    
    try:
        result = supabase.table("user_authentication")\
            .select("subscription_status, usage_count, usage_limit")\
            .eq("email", email)\
            .execute()
        if result.data and len(result.data) > 0:
            return result.data[0]
        else:
            default = {"subscription_status": "free", "usage_count": 0, "usage_limit": 10}
            supabase.table("user_authentication").insert({
                "email": email,
                "subscription_status": default["subscription_status"],
                "usage_count": default["usage_count"],
                "usage_limit": default["usage_limit"]
            }).execute()
            return default
    except Exception as e:
        st.error(f"获取订阅信息失败: {e}")
        return {"subscription_status": "free", "usage_count": 0, "usage_limit": 10}

def increment_usage_count(email: str):
    """使用管理员客户端更新使用次数"""
    supabase = get_supabase_admin_client()
    if supabase is None:
        return False
    user = get_user_subscription(email)
    if user["subscription_status"] == "active":
        return True
    new_count = user["usage_count"] + 1
    try:
        supabase.table("user_authentication")\
            .update({"usage_count": new_count})\
            .eq("email", email)\
            .execute()
        return True
    except Exception as e:
        st.error(f"更新使用次数失败: {e}")
        return False

def can_use_tool(email: str) -> bool:
    """检查用户是否有权限使用工具"""
    user = get_user_subscription(email)
    if user["subscription_status"] == "active":
        return True
    if user["usage_count"] < user["usage_limit"]:
        return True
    else:
        st.warning(f"您本月免费次数已用完（{user['usage_limit']}次）。请升级订阅以继续使用。")
        return False

def get_subscription_display(email: str, lang: str = "zh"):
    """获取用于 UI 显示的订阅信息"""
    user = get_user_subscription(email)
    is_pro = user["subscription_status"] == "active"
    used = user["usage_count"]
    limit = user["usage_limit"]
    if lang == "zh":
        plan_text = "专业版" if is_pro else "免费版"
        remaining_text = "无限次使用" if is_pro else f"本月剩余免费次数: {max(0, limit - used)}"
    else:
        plan_text = "Pro" if is_pro else "Free"
        remaining_text = "Unlimited" if is_pro else f"Free trial remaining: {max(0, limit - used)}"
    return {
        "plan_text": plan_text,
        "remaining_text": remaining_text,
        "is_pro": is_pro,
        "used": used,
        "limit": limit
    }
