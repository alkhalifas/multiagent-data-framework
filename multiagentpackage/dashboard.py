"""
Streamlit Dashboard for MultiAgent Data Exploration
===================================================

This script launches a Streamlit web application that wraps the
multi‑agent framework defined in :mod:`multiagent.agents`. The dashboard
serves two purposes:

* It surfaces proactive insights and patterns by calling
  :meth:`MultiAgentManager.generate_trends` and displaying the results in
  human‑readable form.
* It provides an interactive text box where you can ask natural language
  questions about your data. The appropriate data agents are called under
  the hood, and the answers are displayed in real time.

To run the dashboard:

```bash
streamlit run dashboard.py
```

Make sure you have installed Streamlit and the dependencies listed in
``README.md`` and that your ``OPENAI_API_KEY`` environment variable is set.
"""

import streamlit as st  # type: ignore

from multiagent.agents import create_default_agents


def main() -> None:
    st.set_page_config(page_title="MultiAgent Data Dashboard", layout="wide")
    st.title("Supply Chain, Manufacturing & Sales Insights")

    # Initialise agents lazily when the page is first loaded
    @st.cache_resource(show_spinner=False)
    def get_manager():
        manager, _ = create_default_agents()
        return manager

    manager = get_manager()

    # Sidebar with instructions
    st.sidebar.header("Instructions")
    st.sidebar.write(
        "This dashboard lets you explore your supply chain, manufacturing and "
        "sales data using natural language. Click the **Generate Insights** "
        "button to see recent trends and patterns across all datasets. Then "
        "type a question into the input box to query the data."
    )

    # Main panel
    if st.button("Generate Insights"):
        with st.spinner("Analyzing datasets..."):
            trends = manager.generate_trends()
        st.subheader("Proactive Insights")
        st.text(trends)

    user_question = st.text_input(
        "Ask a question about your data", "", placeholder="e.g. What was the total revenue last month?"
    )
    if user_question:
        with st.spinner("Querying data agents..."):
            answer = manager.ask(user_question)
        st.subheader("Answer")
        st.write(answer)


if __name__ == "__main__":
    main()