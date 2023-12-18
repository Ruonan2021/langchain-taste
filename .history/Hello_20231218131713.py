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
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("Interactive table:")
    #  can pass almost anything to st.write(): text, data, Matplotlib figures, Altair charts, and more. 
    st.write(pd.DataFrame({
        'first column': [1, 2, 3, 4],
        'second column': [10, 20, 30, 40]
    }))

    dataframe = np.random.randn(10, 20)
    st.dataframe(dataframe)

    st.write("Interactive table with highlights")
    # using the Pandas Styler object to highlight some elements in the interactive table.
    dataframe2 = pd.DataFrame(
      np.random.randn(10, 20),
      # Styler 
      columns=('col %d' % i for i in range(20))
    )
    st.dataframe(dataframe2.style.highlight_max(axis=0))

    # static table
    st.write("Static table")
    st.table(dataframe2)

    st.write("Line chart")
    # Line chart
    chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c']
    )
    st.line_chart(chart_data)

    # plot a map
    st.write("Map")
    map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])
    st.map(map_data)

    st.write("# Widgets👋")
    st.write("Slide bar")
    x = st.slider('x')  # 👈 this is a widget
    st.write(x, 'squared is', x * x)
    '''## Note
    Then every time a user interacts with a widget, Streamlit simply reruns your script from top to bottom, assigning the current state of the widget to your variable in the process.'''

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
