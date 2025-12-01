import requests
import pandas as pd
from datetime import datetime
import tabula
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
# -------------------------- 步骤1：新增导入 --------------------------
from email.mime.base import MIMEBase
from email import encoders
# ---------------------------------------------------------------------
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import os

logging.getLogger("org.apache.fontbox").setLevel(logging.ERROR)


current_date = datetime.now().strftime("%y%m%d")
# current_date = '250930'

def check_file_exists():
    filename = f"{current_date}.pdf"
    return os.path.exists(filename)

def save_file(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"文件 '{filename}' 下载成功！")
    else:
        print(f"下载失败，状态码：{response.status_code}")


def create_line_chart(df):
    """Create a line chart from the DataFrame and return it as a bytes object"""
    plt.figure(figsize=(12, 6))
    
    # 筛选需要绘制的列
    plot_columns = [
        col for col in df.columns 
        if (pd.api.types.is_numeric_dtype(df[col]) and 
            not all(df[col].fillna(0) == 0))
    ]
    
    if not plot_columns:
        return None
    
    # 确保索引是datetime类型并按日期升序排列
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            print(f"日期转换错误: {e}")
            return None
    
    # 确保数据按日期升序排列（从左到右时间递增）
    df = df.sort_index()
    
    # 创建图表
    for column in plot_columns:
        line = plt.plot(df.index, df[column], marker='o', label=column)
        
        # 计算变化并标注（从左到右）
        if len(df) >= 2:  # 至少需要两个点才能计算变化
            changes = df[column].diff()  # 计算与前一个点的差值
            for i in range(1, len(df)):  # 从第二个点开始检查
                change = changes.iloc[i]
                if abs(change) > 0.00001:  # 变化超过0.1% 0.001
                    date = df.index[i]
                    y_val = df[column].iloc[i]
                    prev_val = df[column].iloc[i-1]
                    change_pct = (change / prev_val) * 100  # 计算变化百分比
                    
                    # 确定箭头方向和颜色
                    arrow_direction = '↑' if change > 0 else '↓'
                    arrow_color = 'red' if change > 0 else 'blue'
                    bg_color = 'lightcoral' if change > 0 else 'lightblue'
                    
                    # 添加标注
                    plt.annotate(f'{arrow_direction}{abs(change_pct):.2f}%', 
                                xy=(date, y_val),
                                xytext=(0, 15 if change > 0 else -15),
                                textcoords='offset points',
                                ha='center',
                                va='center',
                                bbox=dict(boxstyle='round,pad=0.5', 
                                         fc=bg_color, 
                                         alpha=0.8),
                                arrowprops=dict(arrowstyle='->', 
                                               color=arrow_color,
                                               linewidth=1.5))
    
    plt.title('Japanese Yen TIBOR Rates with Daily Changes')
    plt.ylabel('Rate (%)')
    plt.xlabel('Date')
    
    # 格式化x轴
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, ha='right')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图表
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

def send_email(sender_email, sender_password, recipient_emails, subject, body, chart_data=None, attachments=None):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = ", ".join(recipient_emails)
    msg['Subject'] = subject
    
    css = '''
        <style>
        table{
            border-collapse: collapse;
            width:100%;
            border:1px solid #c6c6c6 !important;
            margin-bottom:20px;
        }
        table th{
            border-collapse: collapse;
            border-right:1px solid #c6c6c6 !important;
            border-bottom:1px solid #c6c6c6 !important;
            background-color:#ddeeff !important; 
            padding:5px 9px;
            font-size:14px;
            font-weight:normal;
            text-align:center;
        }
        table td{
            border-collapse: collapse;
            border-right:1px solid #c6c6c6 !important;
            border-bottom:1px solid #c6c6c6 !important; 
            padding:5px 9px;
            font-size:12px;
            font-weight:normal;
            text-align:center;
            word-break: break-all;
        }
        table tr:nth-child(odd){
            background-color:#fff !important; 
        }
        table tr:nth-child(even){
            background-color: #f8f8f8 !important;
        }
        </style>
    '''
    
    # Add chart if provided
    if chart_data:
        chart_html = '<h3>TIBOR Rates Trend</h3><img src="cid:chart">'
        body = chart_html + body
    
    msg.attach(MIMEText(css + body, 'html'))
    
    # Attach chart image
    if chart_data:
        image = MIMEImage(chart_data.read())
        image.add_header('Content-ID', '<chart>')
        msg.attach(image)
    
    # -------------------------- 步骤2：修正附件处理逻辑 --------------------------
    if attachments:
        for attachment in attachments:
            with open(attachment, 'rb') as file:
                # 创建二进制附件对象
                mime_attachment = MIMEBase('application', 'octet-stream')
                # 设置附件内容
                mime_attachment.set_payload(file.read())
                # Base64编码
                encoders.encode_base64(mime_attachment)
                # 设置附件文件名（仅显示文件名，不含路径）
                filename = os.path.basename(attachment)
                mime_attachment.add_header(
                    'Content-Disposition',
                    f'attachment; filename="{filename}"'
                )
                # 添加附件到邮件
                msg.attach(mime_attachment)
    # -----------------------------------------------------------------------------

    with smtplib.SMTP_SSL('smtp.163.com', 465) as server:
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_emails, msg.as_string())

def calculate_change(df):
    change_list = []
    for column in df.columns:
        try:
            change = df[column].iloc[0] - df[column].iloc[1]
            if abs(change) > 0.001:
                change_list.append(column)
        except (IndexError, TypeError):
            continue
    return change_list

def split_row_to_rows(df):
    if df.empty:
        return pd.DataFrame()
    
    first_row = df.iloc[0].copy()
    first_row = first_row.replace({np.nan: ''})
    first_row = first_row.astype(str)  # 所有值转为字符串，确保后续可调用split()
    split_data = {}

    # 首先处理分割数据
    for col in first_row.index:
        if first_row[col] == '':
            split_data[col] = ['']
        else:
            # 分割字符串并去除空白字符
            split_values = [v.strip() for v in first_row[col].split('\r')]
            split_data[col] = split_values
    
    max_length = max(len(v) for v in split_data.values())
    
    # 统一长度并转换数字
    for col in split_data:
        current_length = len(split_data[col])
        if current_length < max_length:
            split_data[col].extend([''] * (max_length - current_length))
        
        # 尝试将非空字符串转换为float
        converted_values = []
        for val in split_data[col]:
            if val == '':
                converted_values.append(np.nan)
            else:
                try:
                    # 移除可能存在的百分号或其他非数字字符
                    cleaned_val = val.replace('%', '').strip()
                    converted_values.append(float(cleaned_val))
                except (ValueError, AttributeError):
                    # 如果转换失败，保留原始值（通常是日期列）
                    converted_values.append(val)
        split_data[col] = converted_values
    
    new_df = pd.DataFrame(split_data)
    
    # 确保所有数值列都是float类型
    for col in new_df.columns:
        if col != 'Date' and new_df[col].dtype == 'object':
            try:
                new_df[col] = pd.to_numeric(new_df[col], errors='ignore')
            except:
                pass
    
    return new_df

if __name__ == "__main__":
    if not check_file_exists():
        try:
            pdf_url = f"https://www.jbatibor.or.jp/rate/pdf/JAPANESEYENTIBOR{current_date}.pdf"
            filename = f"{current_date}.pdf"
            
            save_file(pdf_url, filename)

            # tables = tabula.read_pdf(
            #     filename,
            #     pages="all",
            #     multiple_tables=False,
            #     lattice=True, 
            #     guess=False
            # )
            
            tables = tabula.read_pdf(
                filename,
                pages="all"
            )

            tables2 = tabula.read_pdf(
                filename,
                pages="all",
                multiple_tables=False,
                stream=True, 
                guess=False,
                pandas_options={'header': 4}
            )

            dfs = [pd.DataFrame(table) for table in tables]
            dfs2 = [pd.DataFrame(table2) for table2 in tables2]

            date_array_by_position = []
            for i, df2 in enumerate(dfs2):
                date_array_by_position = df2.iloc[:, 0].values
            print(date_array_by_position)
            date_list = [s.split()[0] for s in date_array_by_position]
            
            df = pd.concat(dfs, ignore_index=True)
            # print(df.columns)

            # df.insert(loc=0, column='Date', value=['2025-11-04'])
            
            print(df)
            df.columns=[
                'Date',
                '1WEEK',
                '1MONTH',
                '2MONTH',
                '3MONTH',
                '4MONTH',
                '5MONTH',
                '6MONTH',
                '7MONTH',
                '8MONTH',
                '9MONTH',
                '10MONTH',
                '11MONTH',
                '12MONTH']
            df = split_row_to_rows(df)
            # df.insert(0, 'Date', date_list)

            # Convert Date column to datetime if it's not already
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Set Date as index after conversion
            df.set_index('Date', inplace=True)
            df.index.rename('date', inplace=True)

            excel_path = './all_data.xlsx'  # 要发送的Excel文件路径
            
            # 检查Excel文件是否存在并合并数据
            if os.path.exists(excel_path):
                df_new = df.reset_index()
                df_temp = pd.read_excel(excel_path)
                df_temp['date'] = pd.to_datetime(df_temp['date'])
                combined_df = pd.concat([df_temp, df_new], ignore_index=True)
                combined_df.drop_duplicates(subset=['date'], keep='last', inplace=True)
                combined_df.sort_values('date', ascending=False, inplace=True)
                df = combined_df.set_index('date')
                print("数据已与历史记录合并、去重并排序。")
            df.to_excel(excel_path)  # 保存Excel文件
            
            df = df.head(30)
            print(df)
            html_table = df.fillna('').to_html(border=1)
            df.fillna(0, inplace=True)
            
            sender_email = "chengguoyu_82@163.com"
            sender_password = "DUigKtCtMXw34MnB"
            recipient_emails = ["zling@jenseninvest.com","hwang@jenseninvest.com", "yqguo@jenseninvest.com", "13889632722@163.com", "chengguoyu_82@163.com"]
            # recipient_emails = ["wo_oplove@163.com"]
            subject = "Japanese Yen TIBOR"
            
            body = f"<p>Download PDF <a href='{pdf_url}' target='_blank'>click me!</a></p><br/><p>If you would like to see all the data, please check the Excel file in the attachment. </p><br/><div>{html_table}</div><br/>"
            
            change_list = calculate_change(df)
            if change_list:
                change_message = ", ".join(change_list) + " changed by more than 0.1%"
                body = f"**<h3><font color='red'><b>Please note that {change_message}</b></font></h3>**<br/>" + body
            
            # Create and attach chart if there are more than 4 rows
            chart_data = None
            chart_data = create_line_chart(df)
            
            # -------------------------- 步骤3：传入Excel附件路径 --------------------------
            send_email(
                sender_email, 
                sender_password, 
                recipient_emails, 
                subject, 
                body, 
                chart_data, 
                attachments=[excel_path]  # 把Excel文件作为附件传入
            )
            # -----------------------------------------------------------------------------
            print("邮件发送成功，Excel已作为附件添加！")
            
        except Exception as e:
            print("运行时错误:", e)
    else:
        print(f"{current_date}.pdf 已存在，跳过程序执行。")
