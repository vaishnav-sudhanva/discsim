import plotly.express as px

def plot_pie_chart(labels, values, title):
    # Sort values and labels in descending order
    sorted_data = sorted(zip(values, labels), reverse=True)
    sorted_values, sorted_labels = zip(*sorted_data)
    fig = px.pie(names=sorted_labels, values=sorted_values, title=title)
    return fig

def plot_100_stacked_bar_chart(data, x, y, color, title, x_label, y_label):
    fig = px.bar(data, x=x, y=y, color=color, title=title, 
                labels={x: x_label, 'value': y_label},
                barmode='relative')  # This makes it a 100% stacked bar chart
    
    fig.update_layout(legend_title_text='')
    return fig
