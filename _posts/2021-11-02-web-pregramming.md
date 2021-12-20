---
layout: post
title: "Intro to Web programming"
date: 2021-12-20 01:48:45
image: '/images/web.jpg'
description: Trieu Tam
tags:
- Infosec
categories:
- ITMOBRAIN
twitter_text:
---
## Intro to Web-programming
Lập trình web cần học những gì? Đây có lẽ là câu hỏi mà các bạn lập trình viên tương lai đang băn khoăn. Có thể nói, nếu Website là một ngôi nhà thì lập trình web là người đảm nhiệm việc xây dựng nên ngôi nhà đó. Lập trình web đơn giản là công việc sử dụng ngôn ngữ lập trình, công nghệ hỗ trợ để tạo ra một website.

Nghề lập trình web hiện nay đang rất hot và mang lại thu nhập rất cao. Không những thế, đây là nhóm nghề sáng tạo, thu hút rất nhiều tài năng. Gia nhập ngành này là một cơ hội để bạn phát huy hết khả năng của mình dù ở mảng thiết kế hay lập trình thuần túy.

Chúng ta sẽ bắt đầu tìm hiểu về WEBSITE -  thứ chúng ta tương tác hàng ngày khi sử dụng internet.

### Website là gì?
 <img src="/images/Website.jpg">

Website là một tập hợp các trang thông tin có chứa nội dung dạng văn bản, chữ số, âm thanh, hình ảnh, video, v.v... được lưu trữ trên máy chủ (web server) và có thể truy cập từ xa thông qua mạng Internet.

Về mặt tiếng anh, có thể hiểu câu ngữ như sau:<br>
Web = mạng<br>
Site = khu vực, trang

><cite> Như vậy “website” có thể gọi là “trang mạng” hay “trang thông tin điện tử”.</cite>

Nhưng bên cạnh đó còn tồn tại một thuật ngữ khiến chúng ta hay hiểu nhầm với website là “trang web”. Thực chất “trang web” (hay còn gọi là “web page”) là một trang của website, đây là một tài liệu được hiển thị trực tiếp trên trình duyệt như Firefox, Google Chrome, Opera,.. Một website sẽ gồm một hoặc nhiều trang web như vậy. Cùng với nhau, tất cả các Website có thể truy cập công cộng tạo thành World Wide Web (WWW).
### Cấu tạo và hoạt động của website
#### Các thành phần cơ bản tạo nên websites:
- Source Code (mã nguồn): Phần mềm website do các lập trình viên thiết kế xây dựng. Phần này giống như bản thiết kế, vật liệu xây dựng, trang thiết bị nội ngoại thất của ngôi nhà vậy.
- Web hosting (Lưu trữ web):Dùng để lưu trữ mã nguồn. Thành phần này tương tự như mảnh đất để bạn có thể xây dựng ngôi nhà. Là một dịch vụ lưu trữ nằm trên Server hoặc bạn đủ giàu có thể dùng cả Server cho website của bạn. Là trang web của bạn được đặt trên một máy chủ cùng với nhiều trang web khác. Thông thường, các website này chia sẻ chung tài nguyên từ máy chủ như bộ nhớ RAM và CPU.
- Domain (Tên miền): Là địa chỉ của website để các máy tính ở các nơi trỏ tới khi muốn truy cập vào website. Tên miền có vai trò giống như địa chỉ ngôi nhà, dựa vào đó thì người khác mới có thể tìm tới thăm nhà bạn được.
Cách website hoạt động:

#### Mình sẽ mô tả cách hoạt động của website thông qua ví dụ sau:
 <img src="/images/website-hoat-dong-nhu-the-nao.jpg">

1. Đầu tiên người dùng nhập vào trình duyệt một địa chỉ có dạng: https://hocban.vn, thực ra bạn chỉ cần gõ “hocban.vn” là trình duyệt sẽ tự hiểu và đổi thành đường dẫn ở trên.
2. Trình duyệt gửi yêu cầu đến máy chủ DNS.
3. Hệ thống DNS trả kết quả phân tích tên miền trong đường dẫn đã gửi là hocban.vn, nó có địa chỉ máy chủ là 210.211.113.135 (cái này lúc đăng ký người ta đã gán sẵn, máy chủ DNS chỉ cần nhớ thôi).
4. Sau khi nhận được địa chỉ IP – nơi lấy dữ liệu, trình duyệt sẽ tìm đến địa chỉ IP đã nhận – tức máy chủ chứa nội dung website.
5. Máy chủ web nhận được yêu cầu truy xuất nội dung website và nó gửi một tập hợp các file bao gồm HTML, CSS , các tập tin đa phương tiện khác như âm thanh, hình ảnh (nếu có) cho trình duyệt;
6. Trình duyệt “dịch” các file mà máy chủ đã gửi thành trang web mà chúng ta nhìn thấy trên màn hình.

### Nhiệm vụ của lập trình viên Web (full stack).
Người lập trình web sẽ làm nhiệm vụ cơ bản sau:
1. Làm việc với bộ phận thiết kế để nhận được bản vẽ mẫu (bản thiết kế website) hoặc đôi khi tự thiết kế.
2. Sau đó sẽ chuyển bản vẽ thiết kế dạng ảnh trở thành dạng web (HTML / CSS / Javascript) (Front - end)
3. Tiếp đó là viết mã ở bên trong để thực hiện đẩy các thông tin, dữ liệu từ trong cơ sở dữ liệu ra phía khách hàng. (Back End)
4. Thực hiện bảo trì, bảo dưỡng và phát triển thêm các tính năng khác cho website.

### Là một lập trình viên Web thì cần học những gì?
#### 1. Kiến thức nền tảng
Trước hết chúng ta cần hiểu một vài khái niệm như domain, client, server là gì. Sau đó cần tích lũy kiến thức cơ bản về HTML, CSS, Javascript. Đây là cơ sở, nền tảng của một lập trình viên Web bắt buộc phải có để có thể học các kiến thức nâng cao khác. 

Sau khi đã có những kiến thức nền tảng rồi, bạn có thể chọn cho mình 1 trong 2 hướng đi sau:

 <img src="/images/fronback.png">
##### Hướng 1: Lập trình Web Front end
Hướng này chịu trách nhiệm thiết kế và xây dựng giao diện cho các trang web hoặc ứng dụng web để người dùng có thể xem và tương tác trực tiếp trên đó. Tất cả mọi thứ bạn nhìn thấy khi điều hướng trên Internet, từ các font chữ, màu sắc, hình ảnh cho tới các menu xổ xuống và các thanh trượt, là một sự kết hợp của HTML, CSS, và JavaScript được điều khiển bởi trình duyệt máy tính của bạn (Hướng này còn được ví như “phần nổi của tảng bằng chìm”)
Sau khi đã học HTML, CSS, JS vững bạn cần học tiếp các công nghệ:
JQuery: là một thư viện JavaScript thu nhỏ. Có tác dụng giúp tạo ra các tương tác, sự kiện, hiệu ứng trên website… một cách dễ dàng.
CSS và các framework front-end hiện nay phổ biến nhất là Bootstrap giúp hỗ trợ thiết kế website nhanh và chuẩn hơn. Đây là Framework mà hầu hết Front End developer đều cần bạn am hiểu và vận dụng tốt.
Các framework của JavaScript: Có kiến thức và kỹ năng sử dụng thành thạo các Framework của Javascript như AngularJS, Backbone, Ember, ReactJS. Các Frameworks này giúp lập trình viên tiết kiệm được thời gian trong quá trình lập trình, tối ưu hóa và dễ dàng tạo ra các tương tác thân thiện với người dùng.
Các kiến thức về UI (User Interface)  / UX (User Experience). 

##### Hướng 2: Lập trình web Back end
Hướng này bạn sẽ chịu trách nhiệm thiết kế và lập trình phần logic bên trong website để kết nối phần giao diện với cơ sở dữ liệu, giúp cho website sống động và có thể tương tác với người dùng. (Hướng này chính là phần chìm của tảng băng, là những gì người dùng ko thấy trên giao diện web)
Phần này bạn cần học thêm về cách công nghệ như sau:
Ngôn ngữ lập trình chính: Java, PHP, C#, Javascript, Ruby hoặc Python.
Kiến thức về các hệ quản trị cơ sở dữ liệu là vô cùng cần thiết, quan trọng và không thể thiếu như: MS SQL Server, Oracle, MySql, PostgreSQL, MongoDB, ….
Một Framework / công nghệ phù hợp với ngôn ngữ Backend đã chọn: Spring – Java, ASP.NET – C#, Express/NodeJS – Javascript, Laravel – PHP, …
Deploy: Học cấu hình server, domain, cách publish website lên internet...

Đến đây có thể bạn đã có thể tạo ra một website hoàn chỉnh. Nhưng không dừng lại ở đó.
Bạn hoàn toàn có thể kết hợp cả 2 hướng để trở thành LẬP TRÌNH VIÊN FULL STACK.
Lập trình viên full stack là hướng đi cuối cùng mà bạn hướng đến trong lập trình web, cho dù ban đầu bạn có chọn đi hướng front-end hay back-end đi chăng nữa. Lập trình viên full stack là người làm hết từ a-z, từ code giao diện web, cho đến xử lý các thuật toán logic, làm việc với cơ sở dữ liệu để web có thể tương tác được với người dùng. Bên cạnh đó, họ còn phải có kiến thức về bên server và biết cách vận hành, bảo trì cho website  hoạt động một cách trơn chu,...

### Tổng kết
Như vậy, mình đã chia sẻ những kiến thức về website và những kỹ năng, kiến thức cần có để trở thành một lập trình viên web. Lập trình web là công việc không hề khó khăn như trước giờ chúng ta vẫn nghĩ. Hy vọng, qua bài viết trên, các bạn đã rút ra được điều gì đó mới mẻ và có ích. Chúc bạn thành công trong việc tự tạo ra trang web của mình. 
Thanks for reading!
[Nghề lập trình web có dành cho mình không? ](https://niithanoi.edu.vn/nghe-lap-trinh-web.html )
[ Web là gì? Trang web là gì? Có những loại nào?](https://carly.com.vn/blog/website-la-gi/)
