# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import pandas as pd
import numpy as np
import time
import string
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


@st.cache_data
def generate_random_df(row_num: int, col_num: int):
    """_summary_

    Args:
        row_num (int): _description_
        col_num (int): _description_

    Returns:
        _type_: _description_
    """

    col_name_string = string.ascii_lowercase[:col_num]

    chart_data = pd.DataFrame(
        np.random.randn(row_num, col_num),
        columns=list(col_name_string)
    )

    return chart_data


@st.cache_resource
def progress_bar():
    'Starting a long computation...'

    # Add a placeholder
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        # Update the progress bar with each iteration.
        latest_iteration.text(f'Iteration {i+1}')
        bar.progress(i + 1)
        time.sleep(0.1)

    '...and now we\'re done!'


def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    df = pd.DataFrame({
        'first column': [1, 2, 3, 4],
        'second column': [10, 20, 30, 40]
    })

    st.write("Interactive table:")
    #  can pass almost anything to st.write(): text, data, Matplotlib figures, Altair charts, and more.
    st.write(df)

    # dataframe = np.random.randn(10, 20)
    # st.dataframe(dataframe)

    st.write("Interactive table with highlights")
    # using the Pandas Styler object to highlight some elements in the interactive table.
    # dataframe2 = pd.DataFrame(
    #   np.random.randn(10, 20),
    #   # Styler
    #   columns=('col %d' % i for i in range(20))
    # )
    # st.dataframe(dataframe2.style.highlight_max(axis=0))

    # static table
    st.write("Static table")
    # st.table(df)

    st.write("Line chart")
    # Line chart
    # chart_data = pd.DataFrame(
    #  np.random.randn(20, 3),
    #  columns=['a', 'b', 'c']
    # )
    st.line_chart(df)

    # plot a map
    st.write("Map")
    # map_data = pd.DataFrame(
    # np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    # columns=['lat', 'lon'])
    # st.map(map_data)

    st.write("# Widgets👋")
    # slide bar
    st.write("### Slide bar")
    x = st.slider('x', key='slide bar 1')  # 👈 this is a widget
    st.write(x, 'squared is', x * x)
    st.write("Note: \n\nThen every time a user interacts with a widget, Streamlit simply reruns your script from top to bottom, assigning the current state of the widget to your variable in the process.")

    # check box
    st.write("### Check box")

    if st.checkbox('Show dataframe'):
        st.button('Update data')
        left_column, right_column = st.columns(2)
        with left_column:
            st.write('Cached data')
            chart_data = generate_random_df(4, 4)
            chart_data
            progress_bar()
        with right_column:
            st.write('Not Cached data')
            st.write(pd.DataFrame(np.random.randn(
                4, 3), columns=['a', 'b', 'c']))
            'Starting a long computation...'

            # Add a placeholder
            latest_iteration = st.empty()
            bar = st.progress(0)

            for i in range(100):
                # Update the progress bar with each iteration.
                latest_iteration.text(f'Iteration {i+1}')
                bar.progress(i + 1)
                time.sleep(0.1)

            '...and now we\'re done!'

    # select box
    st.write("### Select box")
    option = st.selectbox('Which number do you like best?', df['first column'])
    st.write('You selected: ', option)
    # select box on the side bar
    option2 = st.sidebar.selectbox(
        'Which number do you like best? side', df['first column'])
    st.sidebar.write('You selected side bar: ', option2)

    # layout, column layout
    left_column, right_column = st.columns(2)
    # You can use a column just like st.sidebar:

    left_column.line_chart(df)
    # Or even better, call Streamlit functions inside a "with" block:
    with right_column:
        chosen = st.radio(
            'Sorting hat',
            ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
        st.write(f"You are in {chosen} house!")

    st.write("### Session state")
    st.session_state

    st.write("# RZ's notes👋")

    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        ### Deployment flow
        When making changes in the code, save the script and click 'i' on top right to refresh. After this, each saving script could lead to a page refresh.

        ### Data flow (this might be drawback)
        Streamlit reruns your entire Python script from top to bottom. This can happen in two situations:
        - Whenever you modify your app's source code.
        - Whenever a user interacts with widgets in the app. For example, when dragging a slider, entering text in an input box, or clicking a button.

        ### Build a demo
        ### Want to learn more?
        - Check out [streamlit.io](https://streamlit.io)
        - Jump into our [documentation](https://docs.streamlit.io)
        - Ask a question in our [community
          forums](https://discuss.streamlit.io)
        ### See more complex demos
        - Use a neural net to [analyze the Udacity Self-driving Car Image
          Dataset](https://github.com/streamlit/demo-self-driving)
        - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    """
    )


if __name__ == "__main__":

    run()
