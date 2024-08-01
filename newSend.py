import socket

class LJ29999:
    def __init__(self, ip, port):     #传入两个参数 IP地址和端口号
        self.socket_dashboard = 0
        if port == 29999:              #判断端口号是否为29999
            try:    #异常判断
                #创建套接字
                self.socket_dashboard = socket.socket()
                #连接
                self.socket_dashboard.connect((ip, port))
            except:
                print("连接失败")
        else:
            print("端口输入错误!")

    def  send(self,j):  #发送函数，传入一个要发送的指令
        self.socket_dashboard.send(str.encode(j, 'utf-8'))
        self.WaitReply() #调用反馈函数

    def WaitReply(self): #反馈函数
        #设置大小等待反馈
        data = self.socket_dashboard.recv(1024)
        self.fk29999 = bytes.decode(data, 'utf-8') #将反馈数据存入fk29999方便在主程序中调用

    def close(self):   #关闭端口
        if (self.socket_dashboard != 0):
            self.socket_dashboard.close()


class  LJ30003:
    def __init__(self, ip, port):
        self.socket_feedback = 0
        if port == 30003:              #判断端口号是否为30003
            try:    #异常判断
                self.socket_feedback = socket.socket()
                self.socket_feedback.connect((ip, port))
            except:
                print("连接失败")
        else:
            print("端口输入错误!")

    def send(self,j):
        self.socket_feedback.send(str.encode(j, 'utf-8'))
        self.WaitReply1()
    def WaitReply1(self):
        data = self.socket_feedback.recv(1024)
        self.fk30003 = bytes.decode(data, 'utf-8')
    def close(self):
        if (self.socket_feedback != 0):
            self.socket_feedback.close()


