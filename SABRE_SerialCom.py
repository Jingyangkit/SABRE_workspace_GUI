import serial
import time


ser = serial.Serial("com3", 9600, timeout=100)


class Arduino:
    def SerialComTest(self):
        TestText = 'T' + 'T' + 'T'
        ser.write(TestText.encode('latin_1'))
        FeedBack = ser.readline().decode('utf-8')
        if len(FeedBack) == 3:
            print("We have completed this communication test")
            print(FeedBack)
        else:
            print(len(FeedBack))
            print("This time the test failed, try again")
            while (len(FeedBack) == 0):
                TestText = 'T' + 'T' + 'T'
                ser.write(TestText.encode('latin_1'))
                FeedBack = ser.readline().decode('utf-8')
                if len(FeedBack) != 0:
                    # list_Info = re.split(",", FeedBack, 2)
                    print("We have completed this communication test")
                    print(FeedBack)
    def GrabTask_Close(self):
        TestText = 'G' + 'T' + 'C'
        ser.write(TestText.encode('latin_1'))
        FeedBack = ser.readline().decode('utf-8')
        if len(FeedBack) == 3:
            print("We have completed this GrabClose Task")
            print(FeedBack)
        else:
            print("This time the Grab failed, try again")
            while (len(FeedBack) == 0):
                TestText = 'G' + 'T' + 'C'
                ser.write(TestText.encode('latin_1'))
                FeedBack = ser.readline().decode('utf-8')
                if len(FeedBack) == 3:
                    # list_Info = re.split(",", FeedBack, 2)
                    print("We have completed this GrabClose Task")
                    print(FeedBack)

    def GrabTask_Open(self):
        TestText = 'G' + 'T' + 'O'
        ser.write(TestText.encode('latin_1'))
        FeedBack = ser.readline().decode('utf-8')
        if len(FeedBack) == 3:
            print("We have completed this GrabOpen Task")
            print(FeedBack)
        else:
            print("This time the Grab failed, try again")
            while (len(FeedBack) == 0):
                TestText = 'G' + 'T' + 'O'
                ser.write(TestText.encode('latin_1'))
                FeedBack = ser.readline().decode('utf-8')
                if len(FeedBack) == 3:
                    # list_Info = re.split(",", FeedBack, 2)
                    print("We have completed this GrabOpen Task")
                    print(FeedBack)

    def Bubbling_Open(self, opentime):
        TestText = 'B' + 'O' + 'T'
        ser.write(TestText.encode('latin_1'))
        FeedBack = ser.readline().decode('utf-8')
        if len(FeedBack) == 3:
            print("We have completed this BubblingOpen Task")
            print(FeedBack)
        else:
            print("This time the BubblingOpen failed, try again")
            while (len(FeedBack) == 0):
                TestText = 'B' + 'S' + 'T'
                ser.write(TestText.encode('latin_1'))
                FeedBack = ser.readline().decode('utf-8')
                if len(FeedBack) == 3:
                    # list_Info = re.split(",", FeedBack, 2)
                    print("We have completed this BubblingOpen Task")
                    print(FeedBack)
        time.sleep(opentime)


    def Bubbling_Close(self):
        TestText = 'B' + 'C' + 'T'
        ser.write(TestText.encode('latin_1'))
        FeedBack = ser.readline().decode('utf-8')
        if len(FeedBack) == 3:
            print("We have completed this BubblingClose Task")
            print(FeedBack)
        else:
            print("This time the BubblingClose failed, try again")
            while (len(FeedBack) == 0):
                TestText = 'B' + 'C' + 'T'
                ser.write(TestText.encode('latin_1'))
                FeedBack = ser.readline().decode('utf-8')
                if len(FeedBack) == 3:
                    # list_Info = re.split(",", FeedBack, 2)
                    print("We have completed this BubblingClose Task")
                    print(FeedBack)


    def Start_MagneticField(self, strengthValue, timeSleep):
        TestText = 'V' + 'S' + chr(int(strengthValue))
        ser.write(TestText.encode('latin_1'))
        FeedBack = ser.readline().decode('utf-8')
        if len(FeedBack) == 3:
            print("We have completed this Variable_StrongMagneticField Task")
            print(FeedBack)
        else:
            print("This time the Variable_StrongMagneticField failed, try again")
            while (len(FeedBack) == 0):
                TestText = 'V' + 'S' + chr(int(strengthValue))
                ser.write(TestText.encode('latin_1'))
                FeedBack = ser.readline().decode('utf-8')
                if len(FeedBack) == 3:
                    # list_Info = re.split(",", FeedBack, 2)
                    print("We have completed this Variable_StrongMagneticField Task")
                    print(FeedBack)
        time.sleep(timeSleep)

    def Stop_MagneticField(self):
        TestText = 'V' + 'B' + 'S'
        ser.write(TestText.encode('latin_1'))
        FeedBack = ser.readline().decode('utf-8')
        if len(FeedBack) == 3:
            print("We have completed this Variable_StrongMagneticField Task")
            print(FeedBack)
        else:
            print("This time the Variable_StrongMagneticField failed, try again")
            while (len(FeedBack) == 0):
                TestText = 'V' + 'B' + 'B'
                ser.write(TestText.encode('latin_1'))
                FeedBack = ser.readline().decode('utf-8')
                if len(FeedBack) == 3:
                    # list_Info = re.split(",", FeedBack, 2)
                    print("We have completed this Variable_StrongMagneticField Task")
                    print(FeedBack)




