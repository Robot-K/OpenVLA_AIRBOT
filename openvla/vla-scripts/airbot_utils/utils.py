import subprocess

class CAN_Tools(object):

    @staticmethod
    def check_can_status(interface):
        # 使用 ip link show 命令获取网络接口状态
        result = subprocess.run(['ip', 'link', 'show', interface], capture_output=True, text=True)
        output = result.stdout.strip()

        # 检查输出中是否包含 'UP' 状态
        if 'UP' in output and 'LOWER_UP' in output:
            return True  # 已激活
        else:
            return False  # 未激活

    @staticmethod
    def activate_can_interface(interface, bitrate):
        # 构造要执行的命令
        command = f'sudo ip link set {interface} up type can bitrate {bitrate}'

        # 使用 Popen 执行命令并自动输入密码
        proc = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()

        # 获取命令执行结果
        return_code = proc.returncode
        if return_code == 0:
            return True, stdout.decode()
        else:
            return False, stderr.decode()