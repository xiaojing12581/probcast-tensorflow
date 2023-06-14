import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot(df, quantiles):
    '''
    Plot the target time series and the predicted quantiles.绘制目标时间序列和预测分位数

    Parameters:
    __________________________________
    df: pd.DataFrame.
        Data frame with target time series and predicted quantiles.具有目标时间序列和预测分位数的数据帧

    quantiles: list.
        Quantiles of target time series which have been predicted.已经预测的目标时间序列的五分位数

    Returns:
    __________________________________
    fig: go.Figure.
        Line chart of target time series and predicted quantiles, one subplot for each target.
        目标时间序列和预测分位数的折线图，每个目标一个子图。
    '''
    
    # get the number of predicted quantiles获取预测的分位数
    n_quantiles = len(quantiles)
    
    # get the number of targets获取目标的数量
    n_targets = int((df.shape[1] - 1) / (n_quantiles + 1))
    
    # plot the predicted quantiles for each target绘制每个目标的预测分位数
    fig = make_subplots(
        subplot_titles=['Target ' + str(i + 1) for i in range(n_targets)],
        vertical_spacing=0.15,
        rows=n_targets,
        cols=1
    )
    
    fig.update_layout(#设置图层
        plot_bgcolor='white',#图的背景颜色
        paper_bgcolor='white',#图像的背景颜色
        margin=dict(t=60, b=60, l=30, r=30),#设置图离图像四周的边距
        font=dict(#设置坐标轴标签的字体及颜色
            color='#1b1f24',
            size=8,
        ),
        legend=dict(#设置图例
            traceorder='normal',
            font=dict(#设置图例的字体及颜色
                color='#1b1f24',
                size=10,
            ),
            x=0,
            y=-0.1,
            orientation='h'#如果水平条形图需设置，竖直条形图不用设置
        ),
    )
    
    fig.update_annotations(#注释可以是列表，也可以是单个字符串
        font=dict(#设置注释的字体参数
            color='#1b1f24',
            size=12,
        )
    )
    
    for i in range(n_targets):
        
        fig.add_trace(# 通过add_trace，往图表中加入折线。折线是go.Scatter类
            go.Scatter(
                x=df['time_idx'],# 声明scatter的x轴数据
                y=df['target_' + str(i + 1)],# 声明scatter的y轴数据
                name='Actual',
                legendgroup='Actual',
                showlegend=True if i == 0 else False,
                mode='lines',
                line=dict(
                    color='#afb8c1',
                    width=1
                )
            ),
            row=i + 1,
            col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['time_idx'],
                y=df['target_' + str(i + 1) + '_0.5'],
                name='Median',
                legendgroup='Median',
                showlegend=True if i == 0 else False,
                mode='lines',
                line=dict(
                    width=1,
                    color='rgba(9, 105, 218, 0.5)',
                ),
            ),
            row=i + 1,
            col=1
        )
        
        for j in range(n_quantiles // 2):
            fig.add_trace(
                go.Scatter(
                    x=df['time_idx'],
                    y=df['target_' + str(i + 1) + '_' + str(quantiles[- (j + 1)])],
                    name='q' + format(quantiles[j], '.1%') + ' - ' + 'q' + format(quantiles[- (j + 1)], '.1%'),
                    legendgroup='q' + format(quantiles[j], '.1%') + ' - ' + 'q' + format(quantiles[- (j + 1)], '.1%'),
                    showlegend=False,#设置不显示图例
                    mode='lines',
                    line=dict(
                        color='rgba(9, 105, 218, ' + str(0.1 * (j + 1)) + ')',
                        width=0.1
                    )
                ),
                row=i + 1,
                col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['time_idx'],
                    y=df['target_' + str(i + 1) + '_' + str(quantiles[j])],
                    name='q' + format(quantiles[j], '.1%') + ' - ' + 'q' + format(quantiles[- (j + 1)], '.1%'),
                    legendgroup='q' + format(quantiles[j], '.1%') + ' - ' + 'q' + format(quantiles[- (j + 1)], '.1%'),
                    showlegend=True if i == 0 else False,
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(9, 105, 218, ' + str(0.1 * (j + 1)) + ')',
                    line=dict(
                        color='rgba(9, 105, 218, ' + str(0.1 * (j + 1)) + ')',
                        width=0.1,
                    ),
                ),
                row=i + 1,
                col=1
            )
        #设置x轴的刻度和标签
        fig.update_xaxes(
            title='Time',#设置坐标轴的标签
            color='#424a53',
            tickfont=dict(
                color='#6e7781',
                size=6,
            ),
            linecolor='#eaeef2',
            mirror=True,
            showgrid=False,
            row=i + 1,
            col=1
        )
        #设置y轴的刻度和标签
        fig.update_yaxes(
            title='Value',#设置坐标轴的标签
            color='#424a53',
            tickfont=dict(#设置刻度的字体大小及颜色
                color='#6e7781',
                size=6,
            ),
            linecolor='#eaeef2',
            mirror=True,
            showgrid=False,
            zeroline=False,
            row=i + 1,
            col=1
        )
    
    return fig
