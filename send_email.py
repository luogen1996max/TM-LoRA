import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.header import Header
from email import encoders

def send_email(contain_txt):
    # 配置
    sender = "1005424786@qq.com"           # 发件人QQ邮箱
    password = "jgneojnazrfpbdde"               # QQ邮箱授权码
    receiver = "1005424786@qq.com"            # 收件人邮箱

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = "模型训练完毕"

    # 正文
    body = f"模型已训练完成，请查收附件。\n\n发件人：根深蒂固的小迷妹 \n\n" + contain_txt
    msg.attach(MIMEText(body, "plain", "utf-8"))

    # # 添加附件
    # with open("./generation_time.txt", "rb") as f:
    #     part = MIMEBase("application", "octet-stream")
    #     part.set_payload(f.read())
    #     encoders.encode_base64(part)
    #     part.add_header("Content-Disposition", "attachment", filename="generation_time.txt")
    #     msg.attach(part)

    # 发送
    server = smtplib.SMTP("smtp.qq.com", 587)
    server.starttls()
    server.login(sender, password)
    server.sendmail(sender, [receiver], msg.as_string())
    server.quit()

    print("邮件发送成功")

# send_email()