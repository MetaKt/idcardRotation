# ID Card Rotation
## เกี่ยวกับ Repos นี้
Repos นี้เป็นส่วนหนึ่งของโปรเจค ai ตรวจจับภาพบัตรประชาชนและหมุนให้เป็นหน้าตรง โดยนำโมเดลที่แทรนแล้วในรูปแบบไฟล์ .h5 ด้วย keras เวอร์ชั่น 2 มาปรับใช้กับภาษา Java และ deeplearning4j

ส่วนที่สำคัญจะประกอบไปด้วยสองส่วนหลัก คือ resource Folder และ Main.java

## resource Folder 
โฟลเดอร์นี้จะอยู่ในส่วนของ gitignore ซึ่ง**จำเป็นต้องสร้างเอง** 
โฟลเดอร์นี้เป็นพื้นที่ไว้เก็บรูปภาพบัตรประชาชนเพื่อนำมาคัดแยกประเภท และตัวโมเดล .h5 โดยการสร้าง ในโฟลเดอร์นี้จำเป็นต้องมี
 - ตัวโมเดล .h5ที่ต้องการจะใช้
 - โฟลเดอร์ย่อย data_for_test ที่มีอีกสองโฟลเดอร์ย่อยข้างใน
     - img_cards และ
     - rotated_cards

รูปภาพบัตรประชาชนในรูปแบบของ .jpg ที่ต้องการจะจัดเรียงด้านจะต้องนำมาใส่ในโฟลเดอร์ data_for_test/img_cards ส่วนโฟลเดอร์  data_for_test/rotated_cards จะเป็นพื้นที่ไว้เก็บภาพผลลัพธ์หลังจากโปรแกรม Main.java ได้ตรวจจับด้านของบัตรประชาชนและทำการหมุนด้านให้ถูกต้อง

## Main.java
โปรแกรมนี้คือการนำโมเดล .h5 ที่เทรนด้วย python (3.11.9) และ keras 2 มาปรับใช้เพื่อตรวจจับด้านบัตรประชาชนจากโฟลเดอร์ data_for_test/img_cards

โปรแกรมนี้เขียนด้วย java (19.0.1) โดยใช้ deeplearning4j (1.0.0-M2) และสามารถรันได้ด้วยการใช้ IDE เช่น Intellij 

โดยผลลัพธ์จะ print ออกมาเป็น "IMG: , Predicted Type: , Confidence: " และ "Rotated image saved at: " เพื่อบอกว่ารูปภาพบัตรนี้ ตรวจจับมาในมุมนี้ ค่าความมั่นใจเท่านี้ และเมื่อหมุนเสร็จจะถูกเซฟอยู่ ณ ที่นี้

### Main arguments
เมื่อกดรัน โปรแกรมนี้จะเริ่มจากรับ arguments สามตัว ซึ่งคือ path ของโฟลเดอร์และโมเดล **จำเป็นต้องปรับใช้เอง โดยจำเป็นต้องใส่ path ของทั้งสามตัวในช่อง Program Arugments เมื่อกดเข้าไปใน Edit configurations...**
- args[0] คือ path ของโฟลเดอร์ data_for_test/img_cards
- args[1] คือ path ของโฟลเดอร์ data_for_test/rotated_cards
- args[2] คือ path ของโมเดล model.h5

### classifyImagesInFolder(String imgFolder, String outputFolder, MultiLayerNetwork model)
หลังจาก import โมเดลสำเร็จ จะเป็นการใช้ฟังก์ชัน classifyImagesInFolder 
- เริ่มจากการนำภาพบัตรประชาชนในโฟลเดอร์มาเตรียมพร้อมก่อนเข้าโมเดล
- นำรูปบัตรประชาชนเข้าโมเดล
- ผลลัพธ์จากโมเดลจะถูกนำคำนวณ confidence และไปใช้ต่อเพื่อหมุนภาพบัตรประชาชนให้ถูกต้อง

 Parameters:
- String imgFolder : path ของโฟลเดอร์ที่มีไฟล์ภาพบัตรประชาชน .jpg 
- MultiLayerNetwork model : ตัวโมเดลที่ import เข้ามา
- String outputFolder : path ของโฟลเดอร์ที่มีไว้เก็บภาพ .jpg ที่ทำการหมุนให้ถูกต้องแล้ว

#### preprocessImage(String imagePath)
  - รูปภาพจะถูกนำไปแปลงให้เป็น array ที่มี shape เป็น 128x128x3
  - ทำการ normalize ค่าต่างๆด้วยการหาร 255 (ค่า 255 คือค่าของ pixel สี และนำมาหารเพื่อไม่ให้ตัวเลขใหญ่เกินไปและอยู่ใน range 0-1 )

Parameter:
- String imagePath : path ของรูปภาพ .jpg เดี่ยวๆที่จะนำไปแปลงใส่ array ก่อนจะนำเข้าโมเดล

#### image.permutei(int... var1)
  - ทำการจัดเรียงลำดับ array ให้ตรงตามที่โมเดลต้องการด้วยฟังก์ชั่น image.permutei()

Parameter :
- int... var1 : ลำดับมิติของ array ที่ต้องการจะจัดเรียงใหม่

#### model.output(INDArray input)
  - ฟังก์ชั่นนี้จะคำนวณผลลัพธ์ออกมาในรูปแบบของ one hot
  - โดยค่าผลลัพธ์นี้จะนำมาแสดง คำนวณความน่าจะเป็น และนำไปใช้ต่อในการแยกประเภทของภาพตามมุม (switch case) เพื่อนำไปหมุนในฟังก์ชั่น rotateImage()

Parameter :
- INDArray input : array ที่แปลงมาจากรูปภาพ และตรงตามมิติที่โมเดลต้องการ [1,128,128,3]

#### Map<Integer, String> categoryLabels
  - Hash Map นี้จะจับคู่องศาที่ต้องหมุนของภาพกับเลข index
  - **หากมีการเพิ่มเติมประเภท จำเป็นที่ตัวเลขจะต้องเรียงตามลำดับของโฟลเดอร์ประเภทองศาที่ต้องหมุนของบัตรใน dataset_for_train เมื่อตอนเทรนด้วย python**

#### rotateImage(BufferedImage image, double angle)
  - ฟังก์ชั่นนี้จะนำค่าตามผลลัพธ์ของโมเดลที่แยกเป็นประเภทต่างๆจาก switch case มาหมุนภาพให้อยู่ในมุมที่ถูกต้อง

Parameter : 
- BufferedImage image : รูปที่ต้องการจะทำการหมุน
- double angle : องศาที่ต้องการจะหมุนภาพ (กำหมดไว้ตาม switch case)


