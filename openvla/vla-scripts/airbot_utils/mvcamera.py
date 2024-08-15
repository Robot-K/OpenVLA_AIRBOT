
# basically from /opt/MVS/Samples/64/Python/GrabImage

import sys
import cv2
import numpy as np
import time

sys.path.append("/opt/MVS/Samples/64/Python/MvImport")
from MvCameraControl_class import *

SDKVersion = MvCamera.MV_CC_GetSDKVersion()
print ("SDKVersion[0x%x]" % SDKVersion)

deviceList = MV_CC_DEVICE_INFO_LIST()
tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

# ch:枚举设备 | en:Enum device
ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
if ret != 0:
    print ("enum devices fail! ret[0x%x]" % ret)
    sys.exit()

if deviceList.nDeviceNum == 0:
    print ("find no device!")
    sys.exit()

print ("Find %d devices!" % deviceList.nDeviceNum)

for i in range(0, deviceList.nDeviceNum):
    mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
    if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
        print ("\ngige device: [%d]" % i)
        strModeName = ""
        for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
            strModeName = strModeName + chr(per)
        print ("device model name: %s" % strModeName)

        nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
        nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
        nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
        nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
        print ("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
    elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
        print ("\nu3v device: [%d]" % i)
        strModeName = ""
        for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
            if per == 0:
                break
            strModeName = strModeName + chr(per)
        print ("device model name: %s" % strModeName)

        strSerialNumber = ""
        for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
            if per == 0:
                break
            strSerialNumber = strSerialNumber + chr(per)
        print ("user serial number: %s" % strSerialNumber)

class VideoCapture:
    def __init__(self, index):

        self.open=False

        # ch:创建相机实例 | en:Creat Camera Object
        cam = MvCamera()

        nConnectionNum = index
        
        # ch:选择设备并创建句柄| en:Select device and create handle
        stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            print ("create handle fail! ret[0x%x]" % ret)
            sys.exit()

        # ch:打开设备 | en:Open device
        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print ("open device fail! ret[0x%x]" % ret)
            sys.exit()
        
        # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
        if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize",nPacketSize)
                if ret != 0:
                    print ("Warning: Set Packet Size fail! ret[0x%x]" % ret)
            else:
                print ("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

        # ch:设置触发模式为off | en:Set trigger mode as off
        ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print ("set trigger mode fail! ret[0x%x]" % ret)
            sys.exit()

        # ch:获取数据包大小 | en:Get payload size
        stParam =  MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        
        ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            print ("get payload size fail! ret[0x%x]" % ret)
            sys.exit()
        nPayloadSize = stParam.nCurValue

        #设置曝光为15ms
        nRet = cam.MV_CC_SetFloatValue("ExposureTime", 15000)
        if ret != 0:
            print ("Set ExposureTime fail! ret[0x%x]" % ret)
            sys.exit()

        #设置fps为30
        nRet = cam.MV_CC_SetFloatValue("ResultingFrameRate", 30)
        if ret != 0:
            print ("Set fps fail! ret[0x%x]" % ret)
            sys.exit()

        #设置gamma校正
        nRet = cam.MV_CC_SetBoolValue("GammaEnable", True)
        if nRet != 0:
            print ("Set GammaEnable fail! ret[0x%x]" % ret)

        #设置gamma类型，user：1，sRGB：2
        nRet = cam.MV_CC_SetEnumValue("GammaSelector", 1);
        if nRet != 0:
            print ("Set GammaSelector fail! ret[0x%x]" % ret)

        #设置gamma值，推荐范围0.5-2，1为线性拉伸
        nRet = cam.MV_CC_SetFloatValue("Gamma", 0.7);
        if nRet != 0:
            print ("Set Gamma fail! ret[0x%x]" % ret)

        # ch:开始取流 | en:Start grab image
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            print ("start grabbing fail! ret[0x%x]" % ret)
            sys.exit()
        
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))

        self.data_buf = (c_ubyte * nPayloadSize)()
        self.cam = cam
        self.stOutFrame = stOutFrame
        self.open = True
    
    def __del__(self):
        cam = self.cam
        # ch:停止取流 | en:Stop grab image
        ret = cam.MV_CC_StopGrabbing()
        if ret != 0:
            print ("stop grabbing fail! ret[0x%x]" % ret)
            del data_buf
            sys.exit()

        # ch:关闭设备 | Close device
        ret = cam.MV_CC_CloseDevice()
        if ret != 0:
            print ("close deivce fail! ret[0x%x]" % ret)
            del data_buf
            sys.exit()

        # ch:销毁句柄 | Destroy handle
        ret = cam.MV_CC_DestroyHandle()
        if ret != 0:
            print ("destroy handle fail! ret[0x%x]" % ret)
            del data_buf
            sys.exit()

        del self.data_buf
            
    def read(self):
        cam = self.cam
        stOutFrame = self.stOutFrame
        #transfer the image into rgb format
        
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        nRGBSize = stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight * 3

        stConvertParam = MV_CC_PIXEL_CONVERT_PARAM_EX()
        memset(byref(stConvertParam), 0, sizeof(stConvertParam))
        stConvertParam.nWidth = stOutFrame.stFrameInfo.nWidth
        stConvertParam.nHeight = stOutFrame.stFrameInfo.nHeight
        stConvertParam.pSrcData = stOutFrame.pBufAddr
        stConvertParam.nSrcDataLen = stOutFrame.stFrameInfo.nFrameLen
        stConvertParam.enSrcPixelType = stOutFrame.stFrameInfo.enPixelType
        stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed 
        stConvertParam.pDstBuffer = (c_ubyte * nRGBSize)()
        stConvertParam.nDstBufferSize = nRGBSize

        if ret == 0:
            width = stOutFrame.stFrameInfo.nWidth
            height = stOutFrame.stFrameInfo.nHeight
            ret = cam.MV_CC_ConvertPixelTypeEx(stConvertParam)
            if ret != 0:
                print ("convert pixel fail! ret[0x%x]" % ret)
                sys.exit()

            img_buff = (c_ubyte * stConvertParam.nDstLen)()
            memmove(byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)
            
            img_buff_np = np.frombuffer(img_buff, dtype=np.uint8).reshape((height, width, 3))
            frame = img_buff_np
            frame = cv2.resize(frame, dsize=(640, 480))
            cam.MV_CC_FreeImageBuffer(stOutFrame)

        else:
            print ("no data[0x%x]" % ret)

        ret = True if ret == 0 else False

        return ret, frame

    def isOpened(self):
        return self.open

if __name__ == "__main__":
    cap = VideoCapture(0)
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if ret:
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Error")
            break
        print(time.time()-start_time)
    cap.__del__()
    cv2.destroyAllWindows()