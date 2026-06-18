import base64
from gmssl import sm4
import binascii


class EncryptUtil:
    @staticmethod
    def encrypt(key: str, data: str) -> str:
        """加密"""
        encrypted_data = EncryptUtil.encrypt_sm4(key, data)
        return base64.b64encode(encrypted_data.encode()).decode()

    @staticmethod
    def decrypt(key: str, data: str) -> str:
        """解密"""
        decoded_data = EncryptUtil.base_convert_str(data)
        return EncryptUtil.decrypt_sm4(key, decoded_data)

    @staticmethod
    def encrypt_sm4(key: str, data: str) -> str:
        """SM4 加密"""
        try:
            key_bytes = binascii.unhexlify(key)  # 16 字节密钥
            sm4_cryptor = sm4.CryptSM4()
            sm4_cryptor.set_key(key_bytes, sm4.SM4_ENCRYPT)
            encrypted_bytes = sm4_cryptor.crypt_ecb(data.encode("utf-8"))
            return base64.b64encode(encrypted_bytes).decode()
        except Exception as e:
            raise SecurityError("SM4 加密失败") from e

    @staticmethod
    def decrypt_sm4(key: str, data: str) -> str:
        """SM4 解密"""
        try:
            key_bytes = binascii.unhexlify(key)  # 16 字节密钥
            sm4_cryptor = sm4.CryptSM4()
            sm4_cryptor.set_key(key_bytes, sm4.SM4_DECRYPT)
            encrypted_bytes = base64.b64decode(data)
            decrypted_bytes = sm4_cryptor.crypt_ecb(encrypted_bytes)
            return decrypted_bytes.decode("utf-8")
        except Exception as e:
            raise SecurityError("SM4 解密失败") from e

    @staticmethod
    def base_convert_str(data: str) -> str:
        """Base64 解码为字符串"""
        if data:
            try:
                return base64.b64decode(data.encode()).decode("gbk")
            except Exception:
                return None
        return None


class SecurityError(Exception):
    """自定义安全异常"""
    pass


if __name__ == "__main__":
    key = "f0faa3dac9684f13921aefd14b385914"  # 32 位 HEX 形式密钥
    plaintext = "Hello, SM4 加密!"

    encrypted_text = EncryptUtil.encrypt(key, plaintext)
    print("加密后:", encrypted_text)

    decrypted_text = EncryptUtil.decrypt(key, "UnEyM2FBY2haeXVabHUwOUxWVitSY2I0NXdXMUhwMENpRE1BL0wzd1hkMlV1d1Y0azc2a3Vsd1R4VDBFY2pGenN4cXpEUTlZNlRORWtDSC8rZng0Wmp4c2o2M05TdURXK1pmNEdkZ3JRUEE9")
    print("解密后:", decrypted_text)
