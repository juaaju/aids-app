import smtplib
from email.mime.text import MIMEText

sender = 'example@gmail.com'
receiver = ['example1@gmail.com', 'example2@gmail.com']

with open('email.txt', 'r') as fp:
    # Create a text/plain message
    msg = MIMEText(fp.read())

server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()

# Google App Password, get it from https://myaccount.google.com/apppasswords
server.login(sender, '...')

server.sendmail(sender, receiver, msg.as_string())

print('Email has been sent')