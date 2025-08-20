import streamlit as st
def apply_units(y_values):
    return y_values * 1000 if st.session_state.return_units == 'Nanovolts' else y_values

def get_y_units():
    return 'Voltage (nV)' if st.session_state.return_units == 'Nanovolts' else 'Voltage (μV)'

def style_layout(fig, title):
    fig.update_layout(
        title=title,
        xaxis_title='Time (ms)',
        yaxis_title=get_y_units(),
        width=700, height=450,
        font_family="Times New Roman" if st.session_state.serif_font else "sans-serif",
        font_color="black",
        title_font_family="Times New Roman" if st.session_state.serif_font else "sans-serif",
        font=dict(size=18),
        showlegend=st.session_state.show_legend
    )
    if not st.session_state.auto_y:
        fig.update_layout(yaxis_range=[st.session_state.y_min, st.session_state.y_max])
    return fig
