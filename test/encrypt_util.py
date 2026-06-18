from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import base64

class SecurityException(Exception):
    pass

class EncryptUtil:

    @staticmethod
    def encrypt(key, plaintext):
        """
        加密函数
        :param key: 密钥
        :param plaintext: 需要加密的字符串
        :return: 加密并经过 Base64 编码后的字符串
        """
        try:
            encrypted_text = EncryptUtil.encrypt_sm4(key, plaintext)
            encrypted_bytes = encrypted_text.encode('utf-8')
            encoded_bytes = base64.b64encode(encrypted_bytes)
            return encoded_bytes.decode('utf-8')
        except Exception as e:
            raise SecurityException(f"Encryption failed: {str(e)}")

    @staticmethod
    def decrypt(key, ciphertext):
        """
        解密函数
        :param key: 密钥
        :param ciphertext: 需要解密的字符串
        :return: 解密后的字符串
        """
        try:
            decoded_text = EncryptUtil.base_convert_str(ciphertext)
            return EncryptUtil.decrypt_sm4(key, decoded_text)
        except Exception as e:
            raise SecurityException(f"Decryption failed: {str(e)}")

    @staticmethod
    def encrypt_sm4(key, data):
        """
        SM4 加密函数
        :param key: 密钥
        :param data: 需要加密的数据
        :return: 加密后的字符串
        """
        try:
            key_bytes = bytes.fromhex(key)
            cipher = Cipher(algorithms.SM4(key_bytes), modes.ECB(), backend=default_backend())
            encryptor = cipher.encryptor()
            padder = padding.PKCS7(algorithms.SM4.block_size).padder()
            padded_data = padder.update(data.encode('utf-8')) + padder.finalize()
            encrypted_bytes = encryptor.update(padded_data) + encryptor.finalize()
            return base64.b64encode(encrypted_bytes).decode('utf-8')
        except Exception as e:
            raise SecurityException(f"SM4 encryption failed: {str(e)}")

    @staticmethod
    def decrypt_sm4(key, data):
        """
        SM4 解密函数
        :param key: 密钥
        :param data: 需要解密的数据
        :return: 解密后的字符串
        """
        try:
            key_bytes = bytes.fromhex(key)
            cipher = Cipher(algorithms.SM4(key_bytes), modes.ECB(), backend=default_backend())
            decryptor = cipher.decryptor()
            decoded_bytes = base64.b64decode(data)
            decrypted_bytes = decryptor.update(decoded_bytes) + decryptor.finalize()
            unpadder = padding.PKCS7(algorithms.SM4.block_size).unpadder()
            unpadded_bytes = unpadder.update(decrypted_bytes) + unpadder.finalize()
            return unpadded_bytes.decode('utf-8')
        except Exception as e:
            raise SecurityException(f"SM4 decryption failed: {str(e)}")

    @staticmethod
    def base_convert_str(encoded_str):
        """
        Base64 解码为字符串
        :param encoded_str: Base64 编码的字符串
        :return: 解码后的字符串
        """
        if encoded_str is not None:
            try:
                decoded_bytes = base64.b64decode(encoded_str)
                return decoded_bytes.decode('gbk')
            except UnicodeDecodeError:
                raise SecurityException("Base64 decoding to GBK failed")
        return None