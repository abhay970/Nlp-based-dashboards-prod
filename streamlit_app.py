"""
Cortex Analyst App - SAP HANA Configuration with Salesforce Dremio Integration
============================================================================
This app allows users to interact with their data using natural language.
Uses stored procedures instead of direct API calls.
OPTIMIZED VERSION FOR FASTER RESPONSE TIMES
All data sources now use the unified Dremio procedure.
ENHANCED: Added timestamp conversion for Salesforce date fields
"""
import json
import time
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import streamlit as st
# from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.exceptions import SnowparkSQLException

# YAML file configurations
YAML_CONFIGS = {
    "Salesforce": "SALESFORCEDB.PUBLIC.SALESFORCE_STAGE/salesforceyaml.yaml",
    "Odoo": "ODOO.PUBLIC.ODOO_STAGE/odoo.yaml",
    "SAP": "SAPHANA.PRODUCTSCHEMA.SAPHANA_STAGE/sap.yaml"
}

# Salesforce date/timestamp field patterns
SALESFORCE_DATE_FIELDS = [
    'END_DATE', 'START_DATE', 'CLOSE_DATE', 'CREATED_DATE', 'LAST_MODIFIED_DATE',
    'LAST_ACTIVITY_DATE', 'LAST_VIEWED_DATE', 'LAST_REFERENCED_DATE',
    'SYSTEM_MODSTAMP', 'BIRTHDAY', 'DUE_DATE', 'REMIND_DATE', 'ACTIVITY_DATE',
    'COMPLETION_DATE', 'EXPIRATION_DATE', 'EFFECTIVE_DATE', 'ENROLLMENT_DATE'
]

cnx = st.connection("snowflake")
session = cnx.session()

def convert_salesforce_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Salesforce timestamp fields from Unix milliseconds to readable dates.
    Handles both integer timestamps and None values.
    """
    if df is None or df.empty:
        return df
    
    df_copy = df.copy()
    
    for col in df_copy.columns:
        # Check if column name indicates it's a date/timestamp field
        col_upper = col.upper()
        if any(date_field in col_upper for date_field in SALESFORCE_DATE_FIELDS):
            try:
                # Convert the column, handling None values and non-numeric data
                def convert_timestamp(value):
                    if pd.isna(value) or value is None or value == 'None':
                        return None
                    
                    # Try to convert to integer (handle string numbers)
                    try:
                        timestamp_ms = int(float(value))
                        # Check if it's a reasonable timestamp (between 1970 and 2100)
                        if 0 < timestamp_ms < 4102444800000:  # Jan 1, 2100 in milliseconds
                            # Convert from milliseconds to seconds for datetime
                            timestamp_s = timestamp_ms / 1000
                            return datetime.fromtimestamp(timestamp_s).strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            return value  # Return original if not a valid timestamp
                    except (ValueError, TypeError, OSError):
                        return value  # Return original if conversion fails
                
                df_copy[col] = df_copy[col].apply(convert_timestamp)
                
            except Exception as e:
                # If there's any error with column conversion, leave it as is
                st.warning(f"Could not convert timestamps in column '{col}': {str(e)}")
                continue
    
    return df_copy


def detect_and_convert_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and convert potential timestamp columns even if they don't match known field names.
    This is a more comprehensive approach that looks at the data patterns.
    """
    if df is None or df.empty:
        return df
    
    df_copy = df.copy()
    
    for col in df_copy.columns:
        # Skip if already converted by name-based conversion
        col_upper = col.upper()
        if any(date_field in col_upper for date_field in SALESFORCE_DATE_FIELDS):
            continue
            
        # Check if column contains timestamp-like values
        sample_values = df_copy[col].dropna().head(10)
        if sample_values.empty:
            continue
            
        # Check if values look like Unix timestamps (13 digits for milliseconds)
        timestamp_pattern = re.compile(r'^\d{13}$')
        timestamp_count = sum(1 for val in sample_values if 
                            isinstance(val, (int, float, str)) and 
                            timestamp_pattern.match(str(val)))
        
        # If more than 70% of sample values look like timestamps, convert the column
        if timestamp_count / len(sample_values) > 0.7:
            try:
                def convert_detected_timestamp(value):
                    if pd.isna(value) or value is None or value == 'None':
                        return None
                    
                    try:
                        timestamp_ms = int(float(value))
                        if 1000000000000 <= timestamp_ms <= 4102444800000:  # Reasonable range
                            timestamp_s = timestamp_ms / 1000
                            return datetime.fromtimestamp(timestamp_s).strftime('%Y-%m-%d)
                        else:
                            return value
                    except (ValueError, TypeError, OSError):
                        return value
                
                df_copy[col] = df_copy[col].apply(convert_detected_timestamp)
                st.info(f"🕒 Detected and converted timestamp column: '{col}'")
                
            except Exception as e:
                continue
    
    return df_copy


def main():
    # Initialize session state
    if "messages" not in st.session_state:
        reset_session_state()
    if "selected_yaml" not in st.session_state:
        st.session_state.selected_yaml = "Salesforce"
    
    show_header_and_sidebar()
    
    # Show initial question only once
    if len(st.session_state.messages) == 0 and st.session_state.selected_yaml and "initial_question_asked" not in st.session_state:
        st.session_state.initial_question_asked = True
        process_user_input("What questions can I ask?")
    
    display_conversation()
    handle_user_inputs()
    handle_error_notifications()


def reset_session_state():
    """Reset important session state elements."""
    st.session_state.messages = []
    st.session_state.active_suggestion = None
    st.session_state.warnings = []
    if "initial_question_asked" in st.session_state:
        del st.session_state.initial_question_asked


def show_header_and_sidebar():
    """Display the header and sidebar of the app."""
    st.title("NLP-Based Dashboards")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("Welcome to AI Analyst! Select your data source and ask questions about your data.")
    
    with col2:
        new_yaml_selection = st.selectbox(
            "Select Data Source:",
            options=list(YAML_CONFIGS.keys()),
            index=list(YAML_CONFIGS.keys()).index(st.session_state.selected_yaml),
            key="yaml_selector"
        )
        
        # Handle data source change
        if new_yaml_selection != st.session_state.selected_yaml:
            st.session_state.messages = []
            st.session_state.active_suggestion = None
            st.session_state.warnings = []
            st.session_state.selected_yaml = new_yaml_selection
            if "initial_question_asked" in st.session_state:
                del st.session_state.initial_question_asked
    
    st.info(f"📊 **{st.session_state.selected_yaml}** data source")
    st.divider()


def handle_user_inputs():
    """Handle user inputs from the chat interface."""
    if not st.session_state.selected_yaml:
        st.warning("Please select a data source first.")
        return
    
    user_input = st.chat_input("What is your question?")
    if user_input:
        process_user_input(user_input)
    elif st.session_state.active_suggestion is not None:
        suggestion = st.session_state.active_suggestion
        st.session_state.active_suggestion = None
        process_user_input(suggestion)


def handle_error_notifications():
    """Handle error notifications."""
    if st.session_state.get("fire_API_error_notify"):
        st.toast("An API error has occurred!", icon="🚨")
        st.session_state["fire_API_error_notify"] = False


def process_user_input(prompt: str):
    """Process user input and update the conversation history."""
    # Clear previous warnings
    st.session_state.warnings = []

    # Create user message (hidden from UI)
    new_user_message = {
        "role": "user",
        "content": [{"type": "text", "text": prompt}],
        "hidden": True
    }
    st.session_state.messages.append(new_user_message)
    
    # Prepare messages for API
    messages_for_api = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages
    ]

    # Show analyst response with progress
    with st.chat_message("analyst"):
        with st.spinner("🤔 Analyzing your Data..."):
            response, error_msg = get_analyst_response(messages_for_api)
            
            if error_msg is None:
                analyst_message = {
                    "role": "analyst",
                    "content": response["message"]["content"],
                    "request_id": response["request_id"],
                }
            else:
                analyst_message = {
                    "role": "analyst",
                    "content": [{"type": "text", "text": error_msg}],
                    "request_id": response.get("request_id", "error"),
                }
                st.session_state["fire_API_error_notify"] = True

            if "warnings" in response:
                st.session_state.warnings = response["warnings"]

            st.session_state.messages.append(analyst_message)
            st.rerun()


def display_warnings():
    """Display warnings to the user."""
    for warning in st.session_state.warnings:
        st.warning(warning["message"], icon="⚠️")


def get_analyst_response(messages: List[Dict]) -> Tuple[Dict, Optional[str]]:
    """
    Send chat history to the Cortex Analyst API via stored procedure.
    OPTIMIZED: Improved error handling and response processing.
    """
    selected_yaml_path = YAML_CONFIGS[st.session_state.selected_yaml]
    semantic_model_file = f"@{selected_yaml_path}"
    
    try:
        # Call stored procedure with timeout handling
        result = session.call(
            "CORTEX_ANALYST.CORTEX_AI.CORTEX_ANALYST_API_PROCEDURE",
            messages,
            semantic_model_file
        )
        
        if result is None:
            return {"request_id": "error"}, "❌ No response from Cortex Analyst procedure"
        
        # Parse response
        if isinstance(result, str):
            response_data = json.loads(result)
        else:
            response_data = result
        
        # Handle successful response
        if response_data.get("success", False):
            return_data = {
                "message": response_data.get("analyst_response", {}),
                "request_id": response_data.get("request_id", "N/A"),
                "warnings": response_data.get("warnings", [])
            }
            return return_data, None
        
        # Handle error response
        error_details = response_data.get("error_details", {})
        error_msg = f"""
❌ **Cortex Analyst Error**

**Error Code:** `{error_details.get('error_code', 'N/A')}`  
**Request ID:** `{error_details.get('request_id', 'N/A')}`  
**Status:** `{error_details.get('response_code', 'N/A')}`

**Message:** {error_details.get('error_message', 'No error message provided')}

💡 **Troubleshooting:**
- Verify your {st.session_state.selected_yaml.lower()}.yaml file exists in the stage
- Check database and schema permissions
- Ensure Cortex Analyst is properly configured
        """
        
        return_data = {
            "request_id": response_data.get("request_id", "error"),
            "warnings": response_data.get("warnings", [])
        }
        return return_data, error_msg
        
    except SnowparkSQLException as e:
        error_msg = f"""
❌ **Database Error**

{str(e)}

💡 **Check:**
- Procedure exists: `CORTEX_ANALYST.CORTEX_AI.CORTEX_ANALYST_API_PROCEDURE`
- You have EXECUTE permissions
- YAML file exists in stage
        """
        return {"request_id": "error"}, error_msg
        
    except Exception as e:
        error_msg = f"❌ **Unexpected Error:** {str(e)}"
        return {"request_id": "error"}, error_msg


def display_conversation():
    """Display the conversation history (excluding hidden messages)."""
    for idx, message in enumerate(st.session_state.messages):
        if message.get("hidden", False):
            continue
            
        role = message["role"]
        content = message["content"]
        
        with st.chat_message(role):
            display_message(content, idx)


def display_message(content: List[Dict[str, Union[str, Dict]]], message_index: int):
    """Display a single message content."""
    for item in content:
        if item["type"] == "text":
            st.markdown(item["text"])
        elif item["type"] == "suggestions":
            st.markdown("**💡 Suggested questions:**")
            for suggestion_index, suggestion in enumerate(item["suggestions"]):
                if st.button(
                    suggestion, 
                    key=f"suggestion_{message_index}_{suggestion_index}",
                    type="secondary"
                ):
                    st.session_state.active_suggestion = suggestion
        elif item["type"] == "sql":
            display_sql_query(
                item["statement"], message_index, item.get("confidence")
            )


def modify_salesforce_query(sql: str) -> str:
    """
    Optimize SQL queries by removing 'public' schema from salesforceDb references.
    OPTIMIZED: More efficient regex processing.
    """
    import re
    
    # Single pass with multiple patterns
    patterns = [
        (r'("[sS][aA][lL][eE][sS][fF][oO][rR][cC][eE][dD][bB]")\.("[pP][uU][bB][lL][iI][cC]")\.', r'\1.'),
        (r'\b([sS][aA][lL][eE][sS][fF][oO][rR][cC][eE][dD][bB])\.([pP][uU][bB][lL][iI][cC])\.', r'\1.'),
        (r'("[sS][aA][lL][eE][sS][fF][oO][rR][cC][eE][dD][bB]")\.([pP][uU][bB][lL][iI][cC])\.', r'\1.'),
        (r'\b([sS][aA][lL][eE][sS][fF][oO][rR][cC][eE][dD][bB])\.("[pP][uU][bB][lL][iI][cC]")\.', r'\1.')
    ]
    
    for pattern, replacement in patterns:
        sql = re.sub(pattern, replacement, sql)
    
    return sql


@st.cache_data(show_spinner=False, ttl=300)  # Cache for 5 minutes
def execute_data_procedure(query: str, data_source: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Execute data procedure with caching and optimized error handling.
    OPTIMIZED: Added caching and better error messages.
    All data sources now use the unified Dremio procedure.
    ENHANCED: Added timestamp conversion for Salesforce data.
    """
    try:
        # All data sources use the same unified Dremio procedure
        if data_source == "Salesforce":
            modified_query = modify_salesforce_query(query)
            procedure_call = f"CALL SALESFORCE_DREMIO.SALESFORCE_SCHEMA_DREMIO.dremio_data_procedure('{modified_query}')"
        elif data_source == "Odoo":
            procedure_call = f"CALL SALESFORCE_DREMIO.SALESFORCE_SCHEMA_DREMIO.dremio_data_procedure('{query}')"
        elif data_source == "SAP":
            # SAP also uses the same unified Dremio procedure
            procedure_call = f"CALL SALESFORCE_DREMIO.SALESFORCE_SCHEMA_DREMIO.dremio_data_procedure('{query}')"
        else:
            return None, f"❌ Unknown data source: {data_source}"
        
        df = session.sql(procedure_call).to_pandas()
        
        # Convert timestamps for Salesforce data
        if data_source == "Salesforce" and df is not None and not df.empty:
            # First, convert known date fields
            df = convert_salesforce_timestamps(df)
            # Then, try to detect and convert other potential timestamp columns
            df = detect_and_convert_timestamps(df)
        
        return df, None
        
    except SnowparkSQLException as e:
        error_str = str(e).lower()
        
        # Check for specific error patterns that indicate missing data
        if any(pattern in error_str for pattern in [
            "syntax error", 
            "unexpected 'month'", 
            "unexpected 'year'",
            "unexpected 'day'",
            "invalid date",
            "data not available"
        ]):
            return None, "⚠️ Data not available. Please contact your administrator."
        elif "does not exist" in error_str:
            error_msg = f"❌ **{data_source} Procedure Not Found**\n\nVerify the procedure exists and you have access."
        elif "access denied" in error_str or "insufficient privileges" in error_str:
            error_msg = f"❌ **Permission Denied**\n\nInsufficient privileges for {data_source} procedure."
        else:
            error_msg = f"❌ **{data_source} Error:** {str(e)}"
            
        return None, error_msg
        
    except Exception as e:
        return None, f"❌ **Unexpected Error:** {str(e)}"


def display_sql_confidence(confidence: dict):
    """Display SQL confidence information."""
    if confidence is None:
        return
        
    verified_query_used = confidence.get("verified_query_used")
    with st.popover("🔍 Verified Query Info", help="Query verification details"):
        if verified_query_used is None:
            return
            
        st.write(f"**Name:** {verified_query_used.get('name', 'N/A')}")
        st.write(f"**Question:** {verified_query_used.get('question', 'N/A')}")
        st.write(f"**Verified by:** {verified_query_used.get('verified_by', 'N/A')}")
        
        if 'verified_at' in verified_query_used:
            st.write(f"**Verified at:** {datetime.fromtimestamp(verified_query_used['verified_at'])}")
        
        with st.expander("SQL Query"):
            st.code(verified_query_used.get("sql", "N/A"), language="sql")


def display_sql_query(sql: str, message_index: int, confidence: dict):
    """
    Display SQL query and execute it via appropriate data procedure.
    OPTIMIZED: Streamlined display and execution.
    """
    current_data_source = st.session_state.selected_yaml
    
    # Check if query needs modification
    if current_data_source == "Salesforce":
        modified_sql = modify_salesforce_query(sql)
        query_was_modified = sql != modified_sql
    else:
        modified_sql = sql
        query_was_modified = False

    # Display confidence info if available
    display_sql_confidence(confidence)

    # Execute and display results
    with st.expander("📊 Results", expanded=True):
        with st.spinner(f"⚡ Executing via {current_data_source}..."):
            df, err_msg = execute_data_procedure(sql, current_data_source)
            
            if df is None:
                if "Data not available" in err_msg:
                    st.warning("""
                    ⚠️ **No Data Available**
                    
                    The requested data is not available in the system. 
                    This could be because:
                    - The data hasn't been loaded yet
                    - The time period you requested has no records
                    - The specific records don't exist
                    
                    Please contact your administrator for assistance.
                    """)
                else:
                    st.error(err_msg)
            elif df.empty:
                st.warning("""
                📭 **No Records Found**
                
                Your query executed successfully but returned no data.
                Try adjusting your filters or time period.
                """)
            else:
                # Display results in tabs
                data_tab, chart_tab = st.tabs(["📄 Data", "📈 Chart"])
                
                with data_tab:
                    st.dataframe(df, use_container_width=True)
                    st.caption(f"📊 {len(df)} rows returned")

                with chart_tab:
                    display_charts_tab(df, message_index)


def display_charts_tab(df: pd.DataFrame, message_index: int) -> None:
    """
    Display charts tab with improved performance.
    OPTIMIZED: Better column handling and chart options.
    """
    if len(df.columns) < 2:
        st.info("📊 At least 2 columns required for charts")
        return
    
    # Optimize column selection
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    all_cols = df.columns.tolist()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_col = st.selectbox(
            "X-axis", all_cols, 
            key=f"x_col_select_{message_index}"
        )
    
    with col2:
        available_y_cols = [col for col in all_cols if col != x_col]
        y_col = st.selectbox(
            "Y-axis", available_y_cols,
            key=f"y_col_select_{message_index}"
        )
    
    with col3:
        chart_type = st.selectbox(
            "Chart type",
            ["📈 Line", "📊 Bar", "🔢 Area"],
            key=f"chart_type_{message_index}"
        )
    
    # Create chart based on selection
    try:
        chart_data = df.set_index(x_col)[y_col]
        
        if chart_type == "📈 Line":
            st.line_chart(chart_data)
        elif chart_type == "📊 Bar":
            st.bar_chart(chart_data)
        elif chart_type == "🔢 Area":
            st.area_chart(chart_data)
            
    except Exception as e:
        st.error(f"❌ Chart error: {str(e)}")


if __name__ == "__main__":
    main()
