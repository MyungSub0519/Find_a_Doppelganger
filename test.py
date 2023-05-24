import pymysql
from PIL import Image
import io

# 데이터베이스 연결
db = pymysql.connect(host='localhost', user='root', password='0000', db='testdb', charset='utf8')
cursor = db.cursor()

# 이미지 불러오기
cursor.execute('SELECT img FROM testimg where id=2')
result = cursor.fetchone()  # 이미지 필드 가져오기
image_data = result[0]

# 이미지로 변환
image = Image.open(io.BytesIO(image_data))
image.show('static/images/test.jpg')

# 데이터베이스 연결 종료
cursor.close()
db.close()