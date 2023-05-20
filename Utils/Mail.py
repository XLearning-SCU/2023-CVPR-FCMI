import imaplib
import email
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.encoders import encode_base64
from email.header import Header


class Attachment:
    def __init__(self, part):
        self.part = part

    '''判断是否有附件，并解析（解析email对象的part）
    返回列表（内容类型，大小，文件名，数据流）
    '''

    def parse_attachment(self):
        content_disposition = self.part.get("Content-Disposition", None)
        if content_disposition:
            dispositions = content_disposition.strip().split(";")
            if bool(content_disposition and dispositions[0].lower() == "attachment"):
                file_data = self.part.get_payload(decode=True)
                attachment = dict()
                attachment["content_type"] = self.part.get_content_type()
                attachment["size"] = len(file_data)
                de_name = email.header.decode_header(self.part.get_filename())[0]
                name = de_name[0]
                if de_name[1] is not None:
                    name = str(de_name[0], encoding=de_name[1])
                print(name)
                attachment["name"] = name
                attachment["data"] = file_data
                '''保存附件
                fileobject = open(name, "wb")
                fileobject.write(file_data)
                fileobject.close()
                '''
                return attachment
        return None

    def get_attachment_from_file(self, attachment_file_path):
        self.part = MIMEBase('application', "octet-stream")
        self.part.set_payload(open(attachment_file_path, "rb").read())
        encode_base64(self.part)
        self.part.add_header('Content-Disposition',
                             'attachment; filename="%s"' % str(Header(attachment_file_path, 'utf8')))
        return self.part


class EmailOperator:
    def __init__(self, mail=None):
        self.mail = mail if mail is not None else MIMEMultipart()
        # email.message.Message()

    def get_sender_info(self):
        name = email.utils.parseaddr(self.mail["from"])[0]
        de_name = email.header.decode_header(name)[0]
        if de_name[1] is not None:
            name = str(de_name[0], encoding=de_name[1])
        address = email.utils.parseaddr(self.mail["from"])[1]
        return name, address

    def get_receiver_info(self):
        name = email.utils.parseaddr(self.mail["to"])[0]
        de_name = email.header.decode_header(name)[0]
        if de_name[1] is not None:
            name = str(de_name[0], encoding=de_name[1])
        address = email.utils.parseaddr(self.mail["to"])[1]
        return name, address

    def get_subject_content(self):
        de_content = email.header.decode_header(self.mail['subject'])[0]
        if de_content[1] is not None:
            return str(de_content[0], encoding=de_content[1])
        return de_content[0]

    def as_dict(self):
        attachments = []
        body = None
        html = None
        for part in email.message.Message.walk(self.mail):
            attachment = Attachment(part=part).parse_attachment()
            if attachment:
                attachments.append(attachment)
            elif part.get_content_type() == "text/plain":
                if body is None:
                    body = ""
                body += str(part.get_payload(decode=True), encoding='ISO-8859-9')
            elif part.get_content_type() == "text/html":
                if html is None:
                    html = ""
                html += str(part.get_payload(decode=True), encoding='ISO-8859-9')
        return {
            'subject': self.get_subject_content(),
            'body': body,
            'html': html,
            'from': self.get_sender_info(),
            'to': self.get_receiver_info(),
            'attachments': attachments,
        }

    def set_mail_info(self, receive_user, subject, text, text_type, *attachment_file_paths):
        """
        设置邮件的基本信息（收件人，主题，正文，正文类型html或者plain，可变参数附件路径列表）

        :param receive_user:
        :param subject:
        :param text:
        :param text_type:
        :param attachment_file_paths:
        :return:
        """
        self.mail['To'] = receive_user
        self.mail['Subject'] = subject
        self.mail.attach(MIMEText(text, text_type))
        for attachmentFilePath in attachment_file_paths:
            self.mail.attach(Attachment(part=None).get_attachment_from_file(attachmentFilePath))
        return self

    def reset_mail_info(self):
        """
        重新初始化邮件信息部分

        :return:
        """
        self.mail = MIMEMultipart()

    def add_text_part(self, text, text_type):
        """
        自定义邮件正文信息（正文内容，正文格式html或者plain）

        :param text:
        :param text_type:
        :return:
        """
        self.mail.attach(MIMEText(text, text_type))

    def add_attachment(self, filename, filedata):
        """
        增加附件（以流形式添加，可以添加网络获取等流格式）参数（文件名，文件流）

        :param filename:
        :param filedata:
        :return:
        """
        part = MIMEBase('application', "octet-stream")
        part.set_payload(filedata)
        encode_base64(part)
        part.add_header('Content-Disposition', 'attachment; filename="%s"' % str(Header(filename, 'utf8')))
        self.mail.attach(part)

    def add_part(self, part):
        """
        通用方法添加邮件信息（MIMETEXT，MIMEIMAGE,MIMEBASE...）

        :param part:
        :return:
        """
        self.mail.attach(part)

    def show(self):
        mail_info = self.as_dict()
        print(mail_info['subject'])
        print(mail_info['body'])
        print(mail_info['html'])
        print(mail_info['from'])
        print(mail_info['to'])
        for attachment in mail_info['attachments']:
            file = open(attachment['name'], 'wb')
            file.write(attachment['data'])
            file.close()


class MailReader:
    def __init__(self, username, password, server):
        """
        :param username: 用户名
        :param password: 密码
        :param server: imap服务器
        """
        self.mail = imaplib.IMAP4_SSL(server)
        self.mail.login(username, password)
        self.select("INBOX")

    def show_folders(self):
        """
        :return: 所有文件夹
        """
        return self.mail.list()

    def select(self, selector):
        """
        :param selector:
        :return: 选择收件箱（如“INBOX”，如果不知道可以调用showFolders）
        """
        return self.mail.select(selector)

    def search(self, charset, *criteria):
        """
        搜索邮件(参照RFC文档http://tools.ietf.org/html/rfc3501#page-49)
        
        :param charset:
        :param criteria:
        :return:
        """
        return self.mail.search(charset, *criteria)
        # try:
        #     return self.mail.search(charset, *criteria)
        # except Exception:
        #     self.select("INBOX")
        #     return self.mail.search(charset, *criteria)

    def get_unread_mails(self):
        """
        :return: 所有未读的邮件列表（包含邮件序号）
        """
        # for num in str(rml.get_unread_mails()[1][0], encoding='utf-8').split(' '):
        #     if num != '':
        #         mail_info = rml.get_mail_info(num)
        #         print(mail_info['subject'])
        #         print(mail_info['body'])
        #         print(mail_info['html'])
        #         print(mail_info['from'])
        #         print(mail_info['to'])
        #         # 遍历附件列表
        #         for attachment in mail_info['attachments']:
        #             fileob = open(attachment['name'], 'wb')
        #             fileob.write(attachment['data'])
        #             fileob.close()
        return list(map(self.get_email_format, str(self.search(None, "Unseen")[1][0], encoding='utf-8').split(' ')))

    def get_email_format(self, num):
        """
        以RFC822协议格式返回邮件详情的email对象
        
        :param num:
        :return:
        """
        data = self.mail.fetch(num, 'RFC822')
        if data[0] == 'OK':
            return email.message_from_string(str(data[1][0][1], encoding='utf-8'))
        else:
            return "fetch error"

    def show_unread_mails(self):
        self.select('INBOX')
        for ith, mail in enumerate(self.get_unread_mails()):
            print('----------------------------- Mail {:02d} -----------------------------'.format(ith))
            EmailOperator(mail=mail).show()


class MailSender:
    def __init__(self, user, passwd, smtp, port=0, usettls=False):
        self.mailUser = user
        self.mailPassword = passwd
        self.smtpServer = smtp
        self.smtpPort = port
        self.mailServer = smtplib.SMTP(self.smtpServer, self.smtpPort)
        self.mailServer.ehlo()
        if usettls:
            self.mailServer.starttls()
        self.mailServer.ehlo()
        self.mailServer.login(self.mailUser, self.mailPassword)

    def __del__(self):
        self.mailServer.close()

    def send_mail(self, mail):
        """
        发送邮件
        
        :return:
        """
        if not mail['To']:
            print("没有收件人,请先设置邮件基本信息")
            return
        mail['From'] = self.mailUser
        self.mailServer.sendmail(self.mailUser, mail['To'], mail.as_string())
        print('Sent mail to %s' % mail['To'])


def send(mail_title="input('mail_title: ')", mail_text="input('mail_text: ')"):
    """
    样例输入如下(去除前导空格):
    2279296959@qq.com
    latlvfjooqwweaio(你的授权码)
    1127813917@qq.com
    smtp.qq.com
    Test_测试
    This is a test. 这是一个测试

    :return:
    """
    # mail_address = input('mail_address_sender: ')
    # mail_pwd = input('mail_pwd_sender: ')
    # mail_address2 = input('mail_address_receiver: ')
    # mail_smtp = input('mail_smtp: ')
    # mail_title = input('mail_title: ')
    # mail_text = input('mail_text: ')
    mail_address2 = '2279296959@qq.com'
    mail_pwd = 'lacegkbrwkwvfhfg'
    mail_address = '1127813917@qq.com'
    mail_smtp = 'smtp.qq.com'
    # mail_imap = 'imap.qq.com'

    mail = EmailOperator(). \
        set_mail_info(receive_user=mail_address2,
                      subject=mail_title,
                      text=mail_text,
                      text_type='plain').mail
    mail_sender = MailSender(user=mail_address, passwd=mail_pwd, smtp=mail_smtp, port=25)
    mail_sender.send_mail(mail=mail)


def receive():
    """
    样例输入如下(去除前导空格):
    2279296959@qq.com
    latlvfjooqwweaio(你的授权码)
    imap.qq.com

    :return:
    """
    mail_address = input('mail_address: ')
    mail_pwd = input('mail_pwd: ')
    mail_imap = input('mail_imap: ')

    mail_reader = MailReader(mail_address, mail_pwd, mail_imap)
    mail_reader.show_unread_mails()


if __name__ == '__main__':
    send()
