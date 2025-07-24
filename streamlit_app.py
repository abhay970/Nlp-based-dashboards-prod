"""
Cortex Analyst App - SAP HANA Configuration with Salesforce Dremio Integration
============================================================================
This app allows users to interact with their data using natural language.
Uses stored procedures instead of direct API calls.
OPTIMIZED VERSION FOR FASTER RESPONSE TIMES
All data sources now use the unified Dremio procedure.
ENHANCED: Added timestamp conversion for Salesforce date fields
UPDATED: Added pagination, enhanced charts with aggregation, and improved data display
"""
import json
import time
import re
import math
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

# Pagination settings
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 1000
LARGE_DATASET_THRESHOLD = 1000  # When to automatically enable pagination

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
                            return datetime.fromtimestamp(timestamp_s).strftime('%Y-%m-%d')
                        else:
                            return value
                    except (ValueError, TypeError, OSError):
                        return value
                
                df_copy[col] = df_copy[col].apply(convert_detected_timestamp)
                st.info(f"🕒 Detected and converted timestamp column: '{col}'")
                
            except Exception as e:
                continue
    
    return df_copy


def split_dataframe(input_df: pd.DataFrame, rows: int) -> List[pd.DataFrame]:
    """
    Split dataframe into chunks for pagination.
    
    Args:
        input_df (pd.DataFrame): Input dataframe
        rows (int): Number of rows per chunk
        
    Returns:
        List[pd.DataFrame]: List of dataframe chunks
    """
    df_chunks = [input_df.iloc[i:i + rows] for i in range(0, len(input_df), rows)]
    return df_chunks


def display_pagination_controls(query_key: str, total_records: int, page_size: int, current_page: int):
    """
    Display improved pagination controls with better UX design.
    FIXED: Proper state management and return values.
    """
    total_pages = math.ceil(total_records / page_size) if total_records > 0 else 1
    
    if total_pages <= 1:
        return current_page
    
    # Create a well-designed pagination container
    with st.container():
        # Info section with better formatting
        info_col, settings_col = st.columns([3, 1])
        
        with info_col:
            start_record = (current_page - 1) * page_size + 1
            end_record = min(current_page * page_size, total_records)
            st.markdown(f"""
            **📊 Results Overview**  
            Showing **{start_record:,} - {end_record:,}** of **{total_records:,}** records  
            Page **{current_page}** of **{total_pages}**
            """)
        
        with settings_col:
            # Page size selector with better UX
            current_page_size = st.session_state.get(f"page_size_{query_key}", page_size)
            new_page_size = st.selectbox(
                "📄 Per Page", 
                options=[25, 50, 100, 200, 500, 1000],
                index=[25, 50, 100, 200, 500, 1000].index(current_page_size) if current_page_size in [25, 50, 100, 200, 500, 1000] else 2,
                key=f"page_size_selector_{query_key}",
                help="Number of records to show per page"
            )
            if new_page_size != current_page_size:
                st.session_state[f"page_size_{query_key}"] = new_page_size
                st.session_state[f"current_page_{query_key}"] = 1  # Reset to first page
                st.rerun()
        
        st.divider()
        
        # Navigation controls with better layout and icons
        nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 2, 1, 1])
        
        with nav_col1:
            # First page button
            if st.button("⏮️ First", key=f"first_{query_key}", disabled=current_page <= 1, help="Go to first page"):
                st.session_state[f"current_page_{query_key}"] = 1
                st.rerun()
        
        with nav_col2:
            # Previous page button
            if st.button("⬅️ Prev", key=f"prev_{query_key}", disabled=current_page <= 1, help="Go to previous page"):
                st.session_state[f"current_page_{query_key}"] = current_page - 1
                st.rerun()
        
        with nav_col3:
            # Page number input with better styling
            new_page = st.number_input(
                "🔢 Jump to page",
                min_value=1,
                max_value=total_pages,
                value=current_page,
                key=f"page_input_{query_key}",
                help=f"Enter page number (1-{total_pages})"
            )
            if new_page != current_page:
                st.session_state[f"current_page_{query_key}"] = new_page
                st.rerun()
        
        with nav_col4:
            # Next page button
            if st.button("➡️ Next", key=f"next_{query_key}", disabled=current_page >= total_pages, help="Go to next page"):
                st.session_state[f"current_page_{query_key}"] = current_page + 1
                st.rerun()
        
        with nav_col5:
            # Last page button
            if st.button("⏭️ Last", key=f"last_{query_key}", disabled=current_page >= total_pages, help="Go to last page"):
                st.session_state[f"current_page_{query_key}"] = total_pages
                st.rerun()
        
        # Progress bar to show current position
        progress_value = current_page / total_pages
        st.progress(progress_value, text=f"Page {current_page} of {total_pages}")
        
        # Quick page jumps for large datasets
        if total_pages > 10:
            st.markdown("**Quick Jump:**")
            quick_jump_cols = st.columns(min(5, total_pages))
            
            # Show strategic page numbers for quick access
            quick_pages = []
            if total_pages <= 5:
                quick_pages = list(range(1, total_pages + 1))
            else:
                # Show first, some middle pages, and last
                quick_pages = [1]
                if total_pages > 20:
                    step = total_pages // 4
                    quick_pages.extend([step, step * 2, step * 3])
                elif total_pages > 10:
                    mid = total_pages // 2
                    quick_pages.extend([mid - 1, mid, mid + 1])
                quick_pages.append(total_pages)
                # Remove duplicates and sort
                quick_pages = sorted(list(set(quick_pages)))
            
            for i, page_num in enumerate(quick_pages[:5]):  # Limit to 5 quick jump buttons
                if i < len(quick_jump_cols):
                    with quick_jump_cols[i]:
                        is_current = page_num == current_page
                        button_label = f"{'📍 ' if is_current else ''}{page_num}"
                        if st.button(
                            button_label, 
                            key=f"quick_{query_key}_{page_num}",
                            disabled=is_current,
                            help=f"Jump to page {page_num}"
                        ):
                            st.session_state[f"current_page_{query_key}"] = page_num
                            st.rerun()
    
    return current_page

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
    st.session_state.query_results_cache = {}  # Cache for query results
    st.session_state.current_page = {}  # Current page for each query
    st.session_state.total_records = {}  # Total records for each query
    if "initial_question_asked" in st.session_state:
        del st.session_state.initial_question_asked


def show_header_and_sidebar():
    """Display the header and sidebar of the app."""
    st.title("NLP-Based Dashboards")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("Welcome to AI Analyst! Select your data source and ask questions about your data. Large datasets will automatically use pagination for better performance.")
    
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
    
    # Sidebar with pagination settings
    with st.sidebar:
        st.subheader("Pagination Settings")
        st.selectbox(
            "Default Page Size:",
            options=[25, 50, 100, 200, 500, 1000],
            index=2,  # Default to 100
            key="page_size_setting"
        )
        
        st.divider()
        if st.button("Clear Chat History", use_container_width=True):
            reset_session_state()
    
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
    ENHANCED: Better error handling for user-friendly messages.
    """
    try:
        # All data sources use the same unified Dremio procedure
        if data_source == "Salesforce":
            modified_query = modify_salesforce_query(query)
            procedure_call = f"CALL SALESFORCE_DREMIO.SALESFORCE_SCHEMA_DREMIO.dremio_data_procedure('{modified_query}')"
        elif data_source == "Odoo":
            procedure_call = f"CALL SALESFORCE_DREMIO.SALESFORCE_SCHEMA_DREMIO.dremio_data_procedure('{query}')"
        elif data_source == "SAP":
            procedure_call = f"CALL SALESFORCE_DREMIO.SALESFORCE_SCHEMA_DREMIO.dremio_data_procedure('{query}')"
        else:
            return None, f"❌ Unknown data source: {data_source}"
        
        # Execute the procedure
        result = session.sql(procedure_call)
        
        # Convert to pandas DataFrame
        df = result.to_pandas()
        
        # Check if df is None or empty
        if df is None:
            return None, "⚠️ Data is not available right now. Please try again later or contact your administrator."
        
        # Check if the result is actually an error message (string) instead of DataFrame
        if isinstance(df, str):
            return None, "⚠️ Data is not available right now. Please try again later or contact your administrator."
        
        # Check if DataFrame is empty
        if df.empty:
            return None, "📭 No records found for your query. Please try with different criteria."
        
        # Check for error columns in the DataFrame (like ERROR, RECEIVED_TYPE, DATA columns in your image)
        if 'ERROR' in df.columns or 'RECEIVED_TYPE' in df.columns:
            # This means the procedure returned an error in DataFrame format
            return None, "⚠️ Data is not available right now. Please try again later or contact your administrator."
        
        # Check if the DataFrame contains error-like data
        if len(df.columns) >= 3 and any(col.upper() in ['ERROR', 'RECEIVED_TYPE', 'DATA'] for col in df.columns):
            # Check if the first row contains error information
            first_row = df.iloc[0] if len(df) > 0 else None
            if first_row is not None and any(str(val).lower().startswith('error') for val in first_row.values):
                return None, "⚠️ Data is not available right now. Please try again later or contact your administrator."
        
        # Convert timestamps for Salesforce data
        if data_source == "Salesforce" and df is not None and not df.empty:
            # First, convert known date fields
            df = convert_salesforce_timestamps(df)
            # Then, try to detect and convert other potential timestamp columns
            df = detect_and_convert_timestamps(df)
        
        return df, None
        
    except SnowparkSQLException as e:
        error_str = str(e).lower()
        
        # Enhanced error pattern matching
        if any(pattern in error_str for pattern in [
            "syntax error", 
            "unexpected 'month'", 
            "unexpected 'year'",
            "unexpected 'day'",
            "invalid date",
            "data not available",
            "unexpected data format",
            "no data found",
            "empty result",
            "connection error",
            "timeout"
        ]):
            return None, "⚠️ Data is not available right now. Please try again later or contact your administrator."
        elif "does not exist" in error_str:
            return None, "⚠️ Data is not available right now. Please try again later or contact your administrator."
        elif "access denied" in error_str or "insufficient privileges" in error_str:
            return None, "⚠️ Data is not available right now. Please contact your administrator for access."
        else:
            return None, "⚠️ Data is not available right now. Please try again later or contact your administrator."
            
    except Exception as e:
        # Catch all other exceptions and return user-friendly message
        return None, "⚠️ Data is not available right now. Please try again later or contact your administrator."


def display_sql_confidence(confidence: dict):
    """Display SQL confidence information."""
    if confidence is None:
        return

    verified_query_used = confidence.get("verified_query_used")
    if verified_query_used is None:
        return

    # Removed UI display for verified query info
    # If needed later, you can restore the st.popover block


def display_sql_query(sql: str, message_index: int, confidence: dict):
    """
    Display SQL query and execute it via appropriate data procedure with pagination support.
    ENHANCED: Fixed pagination initialization and state management.
    """
    current_data_source = st.session_state.selected_yaml
    query_key = f"query_{message_index}_{hash(sql)}"
    
    # Initialize pagination state properly
    if f"page_size_{query_key}" not in st.session_state:
        st.session_state[f"page_size_{query_key}"] = st.session_state.get('page_size_setting', DEFAULT_PAGE_SIZE)
    
    if f"current_page_{query_key}" not in st.session_state:
        st.session_state[f"current_page_{query_key}"] = 1
    
    page_size = st.session_state[f"page_size_{query_key}"]
    
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
            df_full, err_msg = execute_data_procedure(sql, current_data_source)
            
            if df_full is None:
                # Show user-friendly error message
                if err_msg:
                    st.warning(err_msg)
                else:
                    st.warning("⚠️ Data is not available right now. Please try again later or contact your administrator.")
                return
            elif df_full.empty:
                st.warning("""
                📭 **No Records Found**
                
                Your query executed successfully but returned no data.
                Try adjusting your filters or time period.
                """)
                return
            else:
                # Additional check to make sure df is actually a DataFrame
                if not isinstance(df_full, pd.DataFrame):
                    st.warning("⚠️ Data is not available right now. Please try again later or contact your administrator.")
                    return
                
                total_records = len(df_full)
                st.session_state[f"total_records_{query_key}"] = total_records
                
                # Always show pagination controls if more than 25 records
                needs_pagination = total_records > 25
                
                if needs_pagination:
                    # Better messaging for large datasets
                    st.info(f"📊 **Dataset** - {total_records:,} records found. Using pagination for optimal performance.")
                    
                    # Get current page and page size from session state
                    current_page = st.session_state[f"current_page_{query_key}"]
                    current_page_size = st.session_state[f"page_size_{query_key}"]
                    
                    # Display improved pagination controls
                    new_page = display_pagination_controls(query_key, total_records, current_page_size, current_page)
                    
                    # Update page if changed
                    if new_page != current_page:
                        st.session_state[f"current_page_{query_key}"] = new_page
                        st.rerun()
                    
                    # Calculate start and end indices for current page
                    start_idx = (current_page - 1) * current_page_size
                    end_idx = min(start_idx + current_page_size, total_records)
                    
                    # Get current page data
                    df_to_display = df_full.iloc[start_idx:end_idx]
                        
                else:
                    # For small datasets, show all data
                    df_to_display = df_full
                    current_page_size = total_records
                    st.success(f"✅ **Complete Dataset** - Showing all {total_records:,} records")
                
                # Display results in tabs
                data_tab, chart_tab = st.tabs(["📄 Data", "📈 Chart"])
                
                with data_tab:
                    # Add export options for better UX
                    if needs_pagination:
                        export_col1, export_col2 = st.columns([3, 1])
                        with export_col2:
                            if st.button("📥 Download Current Page", key=f"download_{query_key}"):
                                csv = df_to_display.to_csv(index=False)
                                st.download_button(
                                    label="💾 Download as CSV",
                                    data=csv,
                                    file_name=f"data_page_{current_page}.csv",
                                    mime="text/csv",
                                    key=f"csv_download_{query_key}"
                                )
                    
                    try:
                        st.dataframe(df_to_display, use_container_width=True, height=400)
                        
                        # Better status information
                        if needs_pagination:
                            current_page = st.session_state[f"current_page_{query_key}"]
                            start_record = (current_page - 1) * current_page_size + 1
                            end_record = min(start_record + len(df_to_display) - 1, total_records)
                            
                            # Status bar with better formatting
                            status_col1, status_col2, status_col3 = st.columns(3)
                            with status_col1:
                                st.metric("📄 Current Page", f"{current_page:,}")
                            with status_col2:
                                st.metric("📊 Records Shown", f"{len(df_to_display):,}")
                            with status_col3:
                                st.metric("🗂️ Total Records", f"{total_records:,}")
                        else:
                            st.caption(f"📊 {len(df_to_display)} rows returned")
                            
                    except Exception as display_error:
                        st.warning("⚠️ Data is not available right now. Please try again later or contact your administrator.")
                        return

                with chart_tab:
                    try:
                        # For charting, use a sample of data to avoid memory issues
                        chart_data = df_to_display
                        
                        # If the current page is too large for charting, take a sample
                        if len(chart_data) > 1000:
                            chart_data = chart_data.sample(n=1000, random_state=42)
                            st.info("📈 Chart shows a random sample of 1,000 records from current page for performance.")
                        
                        display_charts_tab(chart_data, message_index)
                        
                        if needs_pagination:
                            st.caption("📊 Chart shows data from current page only")
                            
                    except Exception as chart_error:
                        st.warning("⚠️ Chart display is not available right now. Please try again later.")

def display_charts_tab(df: pd.DataFrame, message_index: int) -> None:
    """
    Display charts tab with improved performance and aggregation options.
    ENHANCED: Added aggregation methods and better chart handling from Code 1.
    """
    if len(df.columns) < 2:
        st.info("📊 At least 2 columns required for charts")
        return
    
    all_cols_set = set(df.columns)
    col1, col2 = st.columns(2)
    
    x_col = col1.selectbox(
        "X axis", all_cols_set, key=f"x_col_select_{message_index}"
    )
    y_col = col2.selectbox(
        "Y axis",
        all_cols_set.difference({x_col}),
        key=f"y_col_select_{message_index}",
    )
    
    # Add aggregation method selector
    col3, col4 = st.columns(2)
    aggregation_method = col3.selectbox(
        "Aggregation Method",
        options=["sum", "average", "count", "max", "min"],
        index=0,  # Default to "sum"
        key=f"agg_method_{message_index}",
        help="Choose how to aggregate duplicate x-axis values"
    )
    
    chart_type = col4.selectbox(
        "Select chart type",
        options=["Line Chart 📈", "Bar Chart 📊"],
        key=f"chart_type_{message_index}",
    )
    
    try:
        # Clean the data for charting
        chart_df = df[[x_col, y_col]].dropna()
        
        # For numeric y-axis, ensure it's numeric (except for count aggregation)
        if aggregation_method != "count" and chart_df[y_col].dtype == 'object':
            try:
                chart_df[y_col] = pd.to_numeric(chart_df[y_col], errors='coerce')
                chart_df = chart_df.dropna()
            except:
                st.warning(f"Could not convert {y_col} to numeric values for charting")
                return
        
        if len(chart_df) == 0:
            st.warning("No valid data available for charting after cleaning")
            return
        
        # Track if aggregation was applied
        aggregation_applied = False
        
        # Group by x-axis if there are duplicate values (aggregate)
        if chart_df[x_col].duplicated().any():
            aggregation_applied = True
            if aggregation_method == "sum":
                if pd.api.types.is_numeric_dtype(chart_df[y_col]):
                    chart_df = chart_df.groupby(x_col)[y_col].sum().reset_index()
                else:
                    st.warning(f"Cannot sum non-numeric values in {y_col}. Using first occurrence instead.")
                    chart_df = chart_df.drop_duplicates(subset=[x_col])
                    aggregation_applied = False
            elif aggregation_method == "average":
                if pd.api.types.is_numeric_dtype(chart_df[y_col]):
                    chart_df = chart_df.groupby(x_col)[y_col].mean().reset_index()
                else:
                    st.warning(f"Cannot average non-numeric values in {y_col}. Using first occurrence instead.")
                    chart_df = chart_df.drop_duplicates(subset=[x_col])
                    aggregation_applied = False
            elif aggregation_method == "count":
                chart_df = chart_df.groupby(x_col)[y_col].count().reset_index()
            elif aggregation_method == "max":
                if pd.api.types.is_numeric_dtype(chart_df[y_col]):
                    chart_df = chart_df.groupby(x_col)[y_col].max().reset_index()
                else:
                    st.warning(f"Cannot find max of non-numeric values in {y_col}. Using first occurrence instead.")
                    chart_df = chart_df.drop_duplicates(subset=[x_col])
                    aggregation_applied = False
            elif aggregation_method == "min":
                if pd.api.types.is_numeric_dtype(chart_df[y_col]):
                    chart_df = chart_df.groupby(x_col)[y_col].min().reset_index()
                else:
                    st.warning(f"Cannot find min of non-numeric values in {y_col}. Using first occurrence instead.")
                    chart_df = chart_df.drop_duplicates(subset=[x_col])
                    aggregation_applied = False
        
        # Limit chart data points for performance
        if len(chart_df) > 100:
            chart_df = chart_df.head(100)
            st.info("Chart limited to first 100 data points for performance")
        
        if chart_type == "Line Chart 📈":
            st.line_chart(chart_df.set_index(x_col)[y_col])
        elif chart_type == "Bar Chart 📊":
            st.bar_chart(chart_df.set_index(x_col)[y_col])
        
        # Display caption indicating aggregation method used
        if aggregation_applied:
            if aggregation_method == "average":
                st.caption(f"📊 Chart shows **{aggregation_method}** of {y_col} values grouped by {x_col}")
            else:
                st.caption(f"📊 Chart shows **{aggregation_method}** of {y_col} values grouped by {x_col}")
        else:
            st.caption(f"📊 Chart shows {y_col} vs {x_col} (no aggregation needed)")
            
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        st.write("Please try selecting different columns or check your data format.")


if __name__ == "__main__":
    main()
