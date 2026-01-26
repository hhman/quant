from email.mime.text import MIMEText
from email.header import Header
from smtplib import SMTP_SSL
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# qq邮箱smtp服务器
host = "smtp.qq.com"
# account为发件人的qq号码
account = os.getenv("MAIL_ACCOUNT")
# password为qq邮箱的授权码
password = os.getenv("MAIL_PASSWORD")
# 发件人的邮箱
sender = os.getenv("MAIL_SENDER")
# 收件人邮箱（逗号分隔）
receivers = os.getenv("MAIL_RECEIVERS", "").split(",")
# 邮件标题
mail_title = os.getenv("MAIL_TITLE", "量化")


def send_mail(mail_content: str):
    """发送邮件，正文内容由mail_content指定。"""
    for receiver in receivers:
        # ssl登录
        smtp = SMTP_SSL(host)
        # set_debuglevel()是用来调试的。参数值为1表示开启调试模式，参数值为0关闭调试模式
        smtp.set_debuglevel(1)
        smtp.ehlo(host)
        smtp.login(account, password)
        msg = MIMEText(mail_content, "plain", "utf-8")
        msg["Subject"] = Header(mail_title, "utf-8")
        msg["From"] = sender
        msg["To"] = receiver
        smtp.sendmail(sender, receiver, msg.as_string())
        smtp.quit()


if __name__ == "__main__":
    send_mail("量化")
