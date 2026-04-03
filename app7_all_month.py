import requests
import pandas as pd
from datetime import datetime
import tabula
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无显示器环境（CentOS服务器）
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import os

logging.getLogger("org.apache.fontbox").setLevel(logging.ERROR)

COLUMNS = ['1WEEK', '1MONTH', '2MONTH', '3MONTH', '4MONTH', '5MONTH',
           '6MONTH', '7MONTH', '8MONTH', '9MONTH', '10MONTH', '11MONTH', '12MONTH']

current_date = datetime.now().strftime("%y%m%d")


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


def extract_dates_from_stream(filename):
    """用stream模式从PDF中提取日期列表"""
    try:
        tables = tabula.read_pdf(
            filename, pages="all", multiple_tables=False,
            stream=True, guess=False, pandas_options={'header': 4}
        )
        if tables and not tables[0].empty:
            raw_dates = tables[0].iloc[:, 0].values
            # stream模式的第一列可能是 "2026/04/01 0.72191" 这种格式
            dates = [str(s).split()[0] for s in raw_dates]
            return dates
    except Exception as e:
        print(f"stream模式提取日期失败: {e}")
    return None


def parse_pdf(filename):
    """
    解析PDF，自动适配月初（少量数据）和月中/月末（多行数据）的不同格式。
    返回一个包含 Date 列和 13 个利率列的 DataFrame。
    """
    # ---- 方案1: 默认模式（月中/月末，数据>=5行时有效）----
    tables = tabula.read_pdf(filename, pages="all")

    if tables and not tables[0].empty and tables[0].shape[1] == 14:
        print(f"使用默认模式解析，检测到 {tables[0].shape[0]} 行数据")
        df = pd.concat([pd.DataFrame(t) for t in tables], ignore_index=True)
        df.columns = ['Date'] + COLUMNS
        return df

    # ---- 方案2: lattice模式（月初，数据少时有效）----
    print("默认模式未识别到表格，切换到lattice模式...")
    tables = tabula.read_pdf(
        filename, pages="all", lattice=True,
        multiple_tables=False, guess=False, pandas_options={'header': 0}
    )

    if not tables or tables[0].empty:
        raise ValueError(f"无法从 {filename} 中解析出任何表格")

    df = tables[0]
    print(f"lattice模式解析成功，原始shape: {df.shape}")

    # lattice模式有两种情况：
    # A) 13列：列名已是 1WEEK,1MONTH...，数据在第1行，可能有\r
    # B) 14列：第1列是日期标题，第1行是真正的列名，第2行是\r分隔的数据
    if df.shape[1] == 14:
        # 情况B：跳过第1行（列名行），取第2行数据，丢弃第1列（日期标题列）
        data_row = df.iloc[1:, 1:]  # 去掉第一列和第一行
        data_row.columns = COLUMNS
        data_row = data_row.reset_index(drop=True)
        df = data_row
    # 情况A：df已经是13列，列名是 1WEEK...

    # 找到包含数据的行（跳过可能的列名行）
    # 拆分\r分隔的多天数据
    data_row_idx = 0
    first_cell = str(df.iloc[data_row_idx, 0]) if pd.notna(df.iloc[data_row_idx, 0]) else ''

    if '\r' in first_cell:
        num_days = len(first_cell.split('\r'))
        split_data = {}
        for col in df.columns:
            cell = str(df.iloc[data_row_idx][col]) if pd.notna(df.iloc[data_row_idx][col]) else ''
            if cell == '' or cell == 'nan':
                split_data[col] = [np.nan] * num_days
            else:
                values = cell.split('\r')
                parsed = []
                for v in values:
                    v = v.strip()
                    try:
                        parsed.append(float(v))
                    except ValueError:
                        parsed.append(np.nan)
                while len(parsed) < num_days:
                    parsed.append(np.nan)
                split_data[col] = parsed
        df = pd.DataFrame(split_data)
    else:
        # 没有\r，每行就是一天，确保数值类型
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 提取日期
    dates = extract_dates_from_stream(filename)
    if dates and len(dates) >= len(df):
        df.insert(0, 'Date', dates[:len(df)])
    else:
        # 兜底：根据文件名日期往前推算工作日
        file_date = datetime.strptime(current_date, "%y%m%d")
        month_start = file_date.replace(day=1)
        bdays = pd.bdate_range(start=month_start, end=file_date)
        recent_bdays = list(bdays[-len(df):][::-1])
        df.insert(0, 'Date', [d.strftime('%Y/%m/%d') for d in recent_bdays])
        print(f"使用推算的工作日日期: {[d.strftime('%Y/%m/%d') for d in recent_bdays]}")

    df.columns = ['Date'] + COLUMNS
    print(f"最终解析结果: {len(df)} 行数据")
    return df


def create_line_chart(df):
    """Create a line chart from the DataFrame and return it as a bytes object"""
    plt.figure(figsize=(12, 6))

    plot_columns = [
        col for col in df.columns
        if (pd.api.types.is_numeric_dtype(df[col]) and
            not all(df[col].fillna(0) == 0))
    ]

    if not plot_columns:
        return None

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            print(f"日期转换错误: {e}")
            return None

    df = df.sort_index()

    for column in plot_columns:
        plt.plot(df.index, df[column], marker='o', label=column)

        if len(df) >= 2:
            changes = df[column].diff()
            for i in range(1, len(df)):
                change = changes.iloc[i]
                if abs(change) > 0.00001:
                    date = df.index[i]
                    y_val = df[column].iloc[i]
                    prev_val = df[column].iloc[i - 1]
                    change_pct = (change / prev_val) * 100

                    arrow_direction = '↑' if change > 0 else '↓'
                    arrow_color = 'red' if change > 0 else 'blue'
                    bg_color = 'lightcoral' if change > 0 else 'lightblue'

                    plt.annotate(
                        f'{arrow_direction}{abs(change_pct):.2f}%',
                        xy=(date, y_val),
                        xytext=(0, 15 if change > 0 else -15),
                        textcoords='offset points',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.5', fc=bg_color, alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color=arrow_color, linewidth=1.5)
                    )

    plt.title('Japanese Yen TIBOR Rates with Daily Changes')
    plt.ylabel('Rate (%)')
    plt.xlabel('Date')

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, ha='right')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

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

    if chart_data:
        chart_html = '<h3>TIBOR Rates Trend</h3><img src="cid:chart">'
        body = chart_html + body

    msg.attach(MIMEText(css + body, 'html'))

    if chart_data:
        image = MIMEImage(chart_data.read())
        image.add_header('Content-ID', '<chart>')
        msg.attach(image)

    if attachments:
        for attachment in attachments:
            with open(attachment, 'rb') as file:
                mime_attachment = MIMEBase('application', 'octet-stream')
                mime_attachment.set_payload(file.read())
                encoders.encode_base64(mime_attachment)
                fname = os.path.basename(attachment)
                mime_attachment.add_header(
                    'Content-Disposition',
                    f'attachment; filename="{fname}"'
                )
                msg.attach(mime_attachment)

    with smtplib.SMTP_SSL('smtp.163.com', 465, timeout=30) as server:
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


if __name__ == "__main__":
    if not check_file_exists():
        try:
            pdf_url = f"https://www.jbatibor.or.jp/rate/pdf/JAPANESEYENTIBOR{current_date}.pdf"
            filename = f"{current_date}.pdf"

            save_file(pdf_url, filename)

            # 自动适配解析
            df = parse_pdf(filename)

            # 转换日期
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.index.rename('date', inplace=True)

            excel_path = './all_data.xlsx'

            # 合并历史数据
            if os.path.exists(excel_path):
                df_new = df.reset_index()
                df_temp = pd.read_excel(excel_path)
                df_temp['date'] = pd.to_datetime(df_temp['date'])
                combined_df = pd.concat([df_temp, df_new], ignore_index=True)
                combined_df.drop_duplicates(subset=['date'], keep='last', inplace=True)
                combined_df.sort_values('date', ascending=False, inplace=True)
                df = combined_df.set_index('date')
                print("数据已与历史记录合并、去重并排序。")

            # 保存前清理多余列
            if 'Unnamed: 0' in df.columns:
                df.drop(columns=['Unnamed: 0'], inplace=True)

            df.to_excel(excel_path)

            df = df.head(30)
            print('head30:')
            print(df)

            html_table = df.fillna('').to_html(border=1)
            df.fillna(0, inplace=True)

            sender_email = "chengguoyu_82@163.com"
            sender_password = "DUigKtCtMXw34MnB"
            recipient_emails = ["chengguoyu_82@163.com", "zling@jenseninvest.com", "hwang@jenseninvest.com", "yqguo@jenseninvest.com", "13889632722@163.com",]
            #"zling@jenseninvest.com", "hwang@jenseninvest.com", "yqguo@jenseninvest.com", "13889632722@163.com",
            subject = "Japanese Yen TIBOR"

            body = (f"<p>Download PDF <a href='{pdf_url}' target='_blank'>click me!</a></p><br/>"
                    f"<p>If you would like to see all the data, please check the Excel file in the attachment.</p><br/>"
                    f"<div>{html_table}</div><br/>")

            change_list = calculate_change(df)
            if change_list:
                change_message = ", ".join(change_list) + " changed by more than 0.1%"
                body = f"**<h3><font color='red'><b>Please note that {change_message}</b></font></h3>**<br/>" + body

            chart_data = create_line_chart(df)

            print('email ready to send...')
            send_email(
                sender_email,
                sender_password,
                recipient_emails,
                subject,
                body,
                chart_data,
                attachments=[excel_path]
            )
            print("邮件发送成功，Excel已作为附件添加！")

        except Exception as e:
            print("运行时错误:", e)
            import traceback
            traceback.print_exc()
    else:
        print(f"{current_date}.pdf 已存在，跳过程序执行。")
