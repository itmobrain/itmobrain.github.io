---
layout: post
title: "Giới thiệu về Hàm CPP"
date: 2021-11-23 01:48:45
image: '/images/func_Cpp/func.jpeg'
description: Triệu Tâm
tags:
- Programming language

categories:
- ITMOBRAIN
twitter_text:
---
### Giới thiệu về hàm (function) trong C++
Trong bài viết này chúng ta cùng tìm hiểu về hàm và biểu thức hàm thông qua các ví dụ.
Trước tiên, hàm là một khối mã thực hiện một nhiệm vụ cụ thể.
Giả sử chúng ta cần tạo một chương trình để vẽ một hình tròn và tô màu cho nó. Chúng ta có thể tạo hai hàm để giải quyết vấn đề này:
- Hàm để vẽ hình tròn.
- Hàm để tô màu cho hình tròn.
  
Việc chia một vấn đề thành nhiều vấn đề nhỏ hơn giúp chương trình dễ hiểu hơn và có thể tái sử dụng.
Hàm có 2 loại: 
1. Hàm chuẩn, đã được định nghĩa trước ở trong C++.
2. Hàm do người dùng tự định nghĩa.
   
Trong bài này chúng ta sẽ đi sâu vào các hàm do người tự định nghĩa.
### Hàm do người dùng tự định nghĩa 
- C++ cho phép lập trình viên tự định nghĩa hàm của riêng họ.
- Trong hàm sẽ chứa những đoạn mã thực hiện một nhiệm vụ cụ thể nào đó và được đặt tên bởi lập trình viên.
- Khi hàm được gọi từ bất kì phần nào của chương trình, tất cả các đoạn mã nằm trong nó sẽ được thực thi.
  
### Khai báo hàm C++
Cú pháp tổng quát để khai báo một hàm là:
<img src="/images/func_Cpp/Untitled.png">

Và đây là ví dụ về khai báo hàm
<img src="/images/func_Cpp/Untitled1.png">

Ở đây, chúng ta có: 
- Tên của hàm là greet()
- Kiểu trả về của hàm là void
- Dấu ngoặc đơn trống () có nghĩa là không có bất kì một tham số nào được truyền vào
- Phần thân hàm (đoạn mã) sẽ được viết bên trong {}

### Gọi một hàm
Trong chương trình trên chúng ta đã khai báo một hàm có tên greet(). Để sử dụng hàm greet(), chúng ta cần gọi nó.
Và đây là cách chúng ta có thể gọi hàm greet() trên.
<img src="/images/func_Cpp/Untitled2.png">

Cách hoạt động của hàm trong C++
<img src="/images/func_Cpp/Untitled3.png">

Ví dụ 1: Hiển thị văn bản
<img src="/images/func_Cpp/Untitled4.png">

Đầu ra
<img src="/images/func_Cpp/Untitled5.png">


Khai báo hàm với tham số
Như đã đề cập ở trên, một hàm có thể khai báo với các tham số (đối số). Tham số là một giá trị được truyền vào khi khai báo một hàm. 
Ví du, chúng ta cùng xem xét hàm sau:
<img src="/images/func_Cpp/Untitled6.png">

Biến num là tham số hàm với kiểu dữ liệu được truyền vào là int.	
Cách truyền một giá trị cho hàm khi gọi hàm:
<img src="/images/func_Cpp/Untitled7.png">

VÍ dụ 2: Khai báo hàm với nhiều tham số
<img src="/images/func_Cpp/Untitled8.png">

### Đầu ra
<img src="/images/func_Cpp/Untitled9.png">
Trong chương trình trên, chúng ta đã sử dụng một hàm với một tham số có kiểu int và một tham số có kiểu float.
Sau đó chúng ta truyền vào hàm biến num1 và num2 dưới dạng đối số. Các giá trị này được lưu trữ bởi các tham số hàm n1 và n2 tương ứng. 
<img src="/images/func_Cpp/Untitled10.png">

Lưu ý: Kiểu của các đối số được truyền trong khi gọi hàm phải khớp với các tham số tương ứng được xác định trong khai báo hàm.

### Return trong hàm
Trong các chương trình trên chúng ta đã sử dụng kiểu void để khai báo hàm. Điều này có nghĩa là hàm không thể trả về bất kì một giá trị nào. 
Trong C++ thì hàm cũng có thể trả về một giá trị. Để làm điều này chúng ta cần xác định returnType của hàm trong khi khai báo hàm.
Sau đó dùng return  để trả về một giá trị từ hàm.
Ví dụ:
<img src="/images/func_Cpp/Untitled11.png">

Ở đây chúng ta có kiểu dữ liệu của hàm là int thay vì void. Điều này có nghĩa là hàm trả về một giá trị int.
Đoạn mã return(a+b) trả về tổng của hai tham số dưới dạng giá trị của hàm.
Dòng lệnh return thể hiện rằng chức năng đã kết thúc. Bất kì mã nào sau khi return bên trong hàm sẽ không được thực thi.

Ví dụ 3: Cộng hai số
<img src="/images/func_Cpp/Untitled12.png">

### Đầu ra
<img src="/images/func_Cpp/Untitled13.png">

Tròn chương trình trên, hàm add() dùng để tính tổng hai số.
Chúng ta truyền vào trong hàm 2 đối số kiểu int với giá trị 100 và 78 trong khi gọi hàm.
Chúng ta lưu giữ giá trị trả về của hàm trong biến sum, và sau đó in ra nó. 
<img src="/images/func_Cpp/Untitled14.png">

### Nguyên mẫu hàm (function prototype)
Trong C++, mã khai báo hàm phải ở trước lệnh gọi hàm. Tuy nhiên, nếu chúng ta muốn định nghĩa một hàm sau lời gọi hàm, chúng ta cần sử dụng nguyên mẫu hàm.
Ví dụ.
<img src="/images/func_Cpp/Untitled15.png">


Trong đoạn mã trên, nguyên mẫu hàm là: 
<img src="/images/func_Cpp/Untitled16.png">

Điều này cung cấp cho trình biên dịch về tên hàm và các thông số của nó. Đó là lý do vì sao chúng ta có thể sử dụng mã để gọi một hàm trước khi hàm đã được xác định. 
Cú pháp của một nguyên mẫu hàm là:
<img src="/images/func_Cpp/Untitled17.png">

Ví dụ 4: Nguyên mẫu hàm C++
<img src="/images/func_Cpp/Untitled18.png">

output:

<img src="/images/func_Cpp/Untitled19.png">

Chương trình trên gần như giống với ví dụ 3. Điều khác biệt duy nhất ở đây là hàm được định nghĩa sau khi gọi hàm. 
Đó là lý do vì sau chúng ta sử dụng nguyên mẫu hàm trong ví dụ này.

### Lợi ích của việc sử dụng các hàm do người dùng xác định
- Các hàm có thể sử dụng lại. Chúng ta có thể khai báo chúng một lần và sử dụng chúng nhiều lần.
- Các hàm làm cho chương trình dễ dàng hơn vì mỗi tác vụ nhỏ được chia thành một hàm.
- Các hàm giúp đọc dễ hiểu hơn. 

### Các chức năng của thư viện C ++
- Các hàm thư viện là các hàm có sẵn trong lập trình C ++.
- Người lập trình có thể sử dụng các hàm thư viện bằng cách gọi các hàm trực tiếp Họ không cần phải tự viết các hàm.
- Một số chức năng thư viện phổ biến trong C ++ là sqrt(), abs(), isdigit(), vv
- Để sử dụng các hàm thư viện, chúng ta thường cần bao gồm tệp tiêu đề trong đó các hàm thư viện này được định nghĩa.
- Ví dụ, để sử dụng các hàm toán học như sqrt() và abs(), chúng ta cần bao gồm tệp tiêu đề cmath.

Ví dụ 5: Chương trình C++ tìm căn bậc 2 của một số
<img src="/images/func_Cpp/Untitled20.png">

Đầu ra

<img src="/images/func_Cpp/Untitled21.png">

Trong chương trình này, hàm thư viện sqrt() được sử dụng để tính căn bậc hai của một số.
Khai báo hàm của sqrt() được định nghĩa trong cmath tệp tiêu đề. Đó là lý do tại sao chúng ta cần sử dụng mã #include <cmath> để sử dụng hàm sqrt().

### Lời kết
Như vậy là chúng mình đã giới thiệu cụ thể cho các bạn thế nào là hàm trong C++ qua các ví dụ, có những loại hàm gì và chúng được sử dụng như thế nào. 
Hi vọng bài viết này đã mang đến cho các bạn những kiến thức mới mẻ và thú vị. Đừng quên ôn tập lại những gì các bạn đã học được nhé. Good luck!

