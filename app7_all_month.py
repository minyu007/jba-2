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
    """Create a polished line chart from the DataFrame and return it as a bytes object"""

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

    # --- 现代配色 ---
    PALETTE = [
        '#2563EB', '#DC2626', '#059669', '#D97706', '#7C3AED',
        '#DB2777', '#0891B2', '#4F46E5', '#CA8A04', '#0D9488',
        '#E11D48', '#6366F1', '#EA580C',
    ]
    UP_COLOR = '#DC2626'
    DOWN_COLOR = '#059669'
    UP_BG = '#FEE2E2'
    DOWN_BG = '#D1FAE5'
    BG_COLOR = '#FAFBFC'
    GRID_COLOR = '#E5E7EB'

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # --- 绘制线条 ---
    for idx, column in enumerate(plot_columns):
        color = PALETTE[idx % len(PALETTE)]
        ax.plot(
            df.index, df[column],
            color=color, linewidth=2.2, label=column,
            marker='o', markersize=5, markerfacecolor='white',
            markeredgecolor=color, markeredgewidth=1.8,
            zorder=3,
        )

    # --- 涨跌标注（标记所有拐点：值发生变化的点） ---
    if len(df) >= 2:
        # 收集所有标注位置，用于防重叠
        annotations = []
        for idx, column in enumerate(plot_columns):
            for i in range(1, len(df)):
                change = df[column].iloc[i] - df[column].iloc[i - 1]
                prev_val = df[column].iloc[i - 1]
                if prev_val == 0 or abs(change) < 0.00001:
                    continue
                change_pct = (change / prev_val) * 100
                date = df.index[i]
                y_val = df[column].iloc[i]
                annotations.append((date, y_val, change, change_pct, idx))

        # 按 (date, y_val) 排序，交替上下偏移避免重叠
        for ann_idx, (date, y_val, change, change_pct, color_idx) in enumerate(annotations):
            arrow_dir = '▲' if change > 0 else '▼'
            txt_color = UP_COLOR if change > 0 else DOWN_COLOR
            bg = UP_BG if change > 0 else DOWN_BG

            # 根据涨跌方向 + 序号微调偏移，减少重叠
            base_offset_y = 16 if change > 0 else -16
            extra = (ann_idx % 3) * 8
            offset_y = base_offset_y + (extra if change > 0 else -extra)

            ax.annotate(
                f'{arrow_dir}{abs(change_pct):.2f}%',
                xy=(date, y_val),
                xytext=(0, offset_y),
                textcoords='offset points',
                ha='center', va='center',
                fontsize=7.5, fontweight='bold', color=txt_color,
                bbox=dict(boxstyle='round,pad=0.3', fc=bg, ec=txt_color, alpha=0.9, linewidth=0.7),
                arrowprops=dict(arrowstyle='->', color=txt_color, linewidth=1, shrinkA=0, shrinkB=3),
                zorder=5,
            )

    # --- 标题与轴标签 ---
    ax.set_title(
        'Japanese Yen TIBOR Rates',
        fontsize=18, fontweight='bold', color='#1F2937',
        pad=20,
    )
    ax.set_ylabel('Rate (%)', fontsize=12, color='#374151', labelpad=10)
    ax.set_xlabel('Date', fontsize=12, color='#374151', labelpad=10)

    # --- 网格 ---
    ax.grid(True, which='major', axis='y', color=GRID_COLOR, linewidth=0.8, alpha=0.7)
    ax.grid(True, which='major', axis='x', color=GRID_COLOR, linewidth=0.5, alpha=0.4, linestyle='--')

    # --- X轴日期格式 ---
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, ha='right', fontsize=10, color='#6B7280')
    plt.yticks(fontsize=10, color='#6B7280')

    # --- 边框 ---
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#D1D5DB')
        ax.spines[spine].set_linewidth(0.8)

    ax.tick_params(axis='both', which='both', length=0)

    # --- 图例 ---
    legend = ax.legend(
        bbox_to_anchor=(1.02, 1), loc='upper left',
        frameon=True, fancybox=True, shadow=False,
        fontsize=9, title='Tenor', title_fontsize=10,
        edgecolor='#D1D5DB', facecolor='white',
    )
    legend.get_frame().set_alpha(0.95)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close()

    return buf


def build_email_html(body_content, chart_cid=None, report_date=None):
    """构建完整的邮件 HTML，包含现代化样式"""
    date_str = report_date or datetime.now().strftime('%Y-%m-%d')

    chart_section = ''
    if chart_cid:
        chart_section = f'''
        <tr><td style="padding:0 30px 25px;">
            <table width="100%" cellpadding="0" cellspacing="0" style="border:none;">
                <tr><td style="background:#F8FAFC; border-radius:8px; padding:20px; text-align:center; border:none;">
                    <img src="cid:{chart_cid}" style="max-width:100%; height:auto; border-radius:6px;" />
                </td></tr>
            </table>
        </td></tr>'''

    html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/></head>
<body style="margin:0; padding:0; background-color:#F3F4F6; font-family:Arial,Helvetica,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background-color:#F3F4F6; padding:20px 0;">
<tr><td align="center">
<table width="720" cellpadding="0" cellspacing="0" style="background:#FFFFFF; border-radius:12px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,0.06);">

    <!-- Header -->
    <tr><td style="background:linear-gradient(135deg,#1E3A5F 0%,#2563EB 100%); padding:28px 30px;">
        <h1 style="margin:0; color:#FFFFFF; font-size:22px; font-weight:700; letter-spacing:0.3px;">Japanese Yen TIBOR Daily Report</h1>
        <p style="margin:6px 0 0; color:#93C5FD; font-size:13px;">{date_str}</p>
    </td></tr>

    <!-- Chart -->
    {chart_section}

    <!-- Body -->
    <tr><td style="padding:0 30px 30px;">
        {body_content}
    </td></tr>

    <!-- Footer -->
    <tr><td style="padding:20px 30px; background:#F9FAFB; border-top:1px solid #E5E7EB;">
        <p style="margin:0; font-size:11px; color:#9CA3AF; text-align:center;">
            This is an automated report generated from JBA TIBOR data.
            &nbsp;|&nbsp; Data source: <a href="https://www.jbatibor.or.jp/" style="color:#6B7280;">jbatibor.or.jp</a>
        </p>
    </td></tr>

</table>
</td></tr></table>
</body></html>'''
    return html


def build_data_table_html(df):
    """将 DataFrame 转换为现代化 HTML 表格（使用 inline style，兼容邮件客户端）
    自动过滤全为 NaN/0 的列，数值格式化为 5 位小数并右对齐"""

    # 过滤掉全为 NaN 或全为 0 的列
    visible_cols = [
        col for col in df.columns
        if df[col].notna().any() and not (df[col].fillna(0) == 0).all()
    ]

    font_family = "'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"
    mono_font = "'SF Mono',Menlo,Consolas,'Liberation Mono',monospace"

    th_style = (
        f'padding:12px 14px; font-size:13px; font-weight:700; color:#1E3A5F; '
        f'text-align:center; border-bottom:2px solid #2563EB; background-color:#EFF6FF; '
        f'font-family:{font_family}; letter-spacing:0.4px; text-transform:uppercase;'
    )
    td_style_num = (
        f'padding:10px 14px; font-size:13px; text-align:right; '
        f'border-bottom:1px solid #E5E7EB; color:#1F2937; '
        f'font-family:{mono_font}; letter-spacing:0.3px;'
    )
    td_style_date = (
        f'padding:10px 14px; font-size:13px; text-align:left; '
        f'border-bottom:1px solid #E5E7EB; font-weight:600; color:#1E3A5F; '
        f'white-space:nowrap; font-family:{font_family};'
    )

    header = '<tr>'
    header += f'<th style="{th_style} text-align:left;">Date</th>'
    for col in visible_cols:
        header += f'<th style="{th_style}">{col}</th>'
    header += '</tr>'

    rows = ''
    for i, (idx, row) in enumerate(df.iterrows()):
        bg = '#FFFFFF' if i % 2 == 0 else '#F8FAFC'
        hover = 'border-left:3px solid transparent;'
        row_style = f'background-color:{bg}; {hover}'
        date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
        rows += f'<tr style="{row_style}">'
        rows += f'<td style="{td_style_date}">{date_str}</td>'
        for col in visible_cols:
            val = row[col]
            cell = f'{val:.5f}' if pd.notna(val) and val != 0 else '-'
            rows += f'<td style="{td_style_num}">{cell}</td>'
        rows += '</tr>'

    table_html = f'''
    <table cellpadding="0" cellspacing="0" style="width:100%; border-collapse:collapse; border-radius:8px; overflow:hidden; border:1px solid #E5E7EB;">
        <thead>{header}</thead>
        <tbody>{rows}</tbody>
    </table>'''
    return table_html


def send_email(sender_email, sender_password, recipient_emails, subject, body, chart_data=None, attachments=None):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = ", ".join(recipient_emails)
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'html'))

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

    import time
    for attempt in range(3):
        try:
            with smtplib.SMTP_SSL('smtp.163.com', 465, timeout=60) as server:
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, recipient_emails, msg.as_string())
            return  # 发送成功，直接返回
        except (smtplib.SMTPServerDisconnected, TimeoutError, OSError) as e:
            print(f"第{attempt + 1}次发送失败: {e}")
            if attempt < 2:
                print(f"等待10秒后重试...")
                time.sleep(10)
            else:
                raise  # 3次都失败，抛出异常


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

            df.fillna(0, inplace=True)

            sender_email = "chengguoyu_82@163.com"
            sender_password = "DUigKtCtMXw34MnB"
            recipient_emails = ["chengguoyu_82@163.com"]
            # recipient_emails = ["chengguoyu_82@163.com", "zling@jenseninvest.com", "hwang@jenseninvest.com", "yqguo@jenseninvest.com", "13889632722@163.com",]
            #"zling@jenseninvest.com", "hwang@jenseninvest.com", "yqguo@jenseninvest.com", "13889632722@163.com",
            subject = "Japanese Yen TIBOR"

            # --- 组装邮件正文 ---
            # 变动提醒
            change_list = calculate_change(df)
            alert_html = ''
            if change_list:
                change_message = ", ".join(change_list)
                alert_html = f'''
                <table cellpadding="0" cellspacing="0" style="width:100%; margin-bottom:20px; border:none;">
                <tr><td style="background:#FEF2F2; border-left:4px solid #DC2626; border-radius:6px; padding:14px 18px; border:none;">
                    <p style="margin:0; font-size:14px; color:#991B1B; font-weight:600;">
                        &#9888; Rate Change Alert
                    </p>
                    <p style="margin:6px 0 0; font-size:13px; color:#B91C1C;">
                        {change_message} changed by more than 0.1%
                    </p>
                </td></tr></table>'''

            # 操作按钮
            actions_html = f'''
            <table cellpadding="0" cellspacing="0" style="width:100%; margin-bottom:24px; border:none;">
            <tr>
                <td style="border:none;">
                    <a href="{pdf_url}" target="_blank"
                       style="display:inline-block; padding:10px 22px; background:#2563EB; color:#FFFFFF;
                              font-size:13px; font-weight:600; text-decoration:none; border-radius:6px;
                              letter-spacing:0.3px;">
                        &#128196; Download PDF
                    </a>
                </td>
            </tr>
            <tr><td style="padding-top:10px; font-size:12px; color:#6B7280; border:none;">
                The full dataset is included as an Excel attachment.
            </td></tr>
            </table>'''

            # 数据表格标题
            table_title = '''
            <p style="font-size:15px; font-weight:600; color:#1F2937; margin:0 0 12px;">
                Recent 30 Days Data
            </p>'''

            # 数据表格
            html_table = build_data_table_html(df)

            body_content = alert_html + actions_html + table_title + html_table

            chart_data = create_line_chart(df)
            report_date = df.index[0].strftime('%Y-%m-%d') if len(df) > 0 else None
            email_html = build_email_html(
                body_content,
                chart_cid='chart' if chart_data else None,
                report_date=report_date,
            )

            print('email ready to send...')
            send_email(
                sender_email,
                sender_password,
                recipient_emails,
                subject,
                email_html,
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
