import pymysql

# # 打开数据库连接,参数1:主机名或IP；参数2：用户名；参数3：密码；参数4：数据库名称
# db = pymysql.connect(host="localhost", user="root", password="123456", database="ais")
# # 使用 cursor() 方法创建一个游标对象 cursor
# cursor = db.cursor()
# # 使用 execute()  方法执行 SQL 查询
# cursor.execute("SELECT VERSION()")
#
# # 如果数据表已经存在使用 execute() 方法删除表。
# cursor.execute("DROP TABLE IF EXISTS ais")
#
# # 创建数据表SQL语句
# sql = """CREATE TABLE EMPLOYEE (
#          FIRST_NAME  CHAR(20) NOT NULL,
#          LAST_NAME  CHAR(20),
#          AGE INT,
#          SEX CHAR(1),
#          INCOME FLOAT )"""
#
# cursor.execute(sql)
#
# # 使用 fetchone() 方法获取单条数据.
# data = cursor.fetchone()
# print("Database version : %s " % data)
# 关闭数据库连接
# db.close()


from PIL import Image, ImageOps

def load_image_with_exif_rotation(image_path):
    image = Image.open(image_path)
    # 自动应用 EXIF 方向标签
    image = ImageOps.exif_transpose(image)
    return image

# 示例
image = load_image_with_exif_rotation("D:\Files\Qt_Projects\LabelTool\Annotation\Annotation\images\img/vit\GBC23B0.jpg")
image.show()  # 此时方向已自动修正

# 手动旋转 90 度（顺时针）
rotated_image = image.rotate(-180, expand=True)

# 保存时清除 EXIF 方向标签
rotated_image.save("D:\Files\Qt_Projects\LabelTool\Annotation\Annotation\images\img/vit\GBC23B0.jpg", exif=b"")