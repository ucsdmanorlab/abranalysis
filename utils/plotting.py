def apply_units(y_values, output_settings):
    return y_values * 1000 if output_settings.return_units == 'Nanovolts' else y_values

def get_y_units(output_settings):
    return 'Voltage (nV)' if output_settings.return_units == 'Nanovolts' else 'Voltage (μV)'

def style_layout(fig, title, output_settings):
    fig.update_layout(
        title=title,
        xaxis_title='Time (ms)',
        yaxis_title=get_y_units(output_settings),
        width=700, height=450,
        font_family="Times New Roman" if output_settings.serif_font else "sans-serif",
        font_color="black",
        title_font_family="Times New Roman" if output_settings.serif_font else "sans-serif",
        font=dict(size=18),
        showlegend=output_settings.show_legend
    )
    if not output_settings.auto_y:
        fig.update_layout(yaxis_range=[output_settings.y_min, output_settings.y_max])
    return fig
