---
layout: post
title: "Intro to Graph Neutral Network"
date: 2022-1-9 01:48:45
image: "/images/gnn/gnn.jpeg"
description: ManhLab
tags:
- Infosec
categories:
- ITMOBRAIN
twitter_text:
---
### Why Graphs? 
><cite>Graphs are a general language for describing and analyzing entities with relations/interactions</cite>

><cite>Complex domains have a rich relational structure, which can be represented as a relational graph. By explicitly modeling relationships we achieve better performance!</cite>


Gần đây, Graph Neural Network (GNN) ngày càng trở nên phổ biến trong nhiều lĩnh vực khác nhau, bao gồm phân tích dữ liệu mạng xã hội, trích xuất kiến thức, hệ thống gợi ý và thậm chí cả phân tích cuộc sống. Sức mạnh của GNN trong việc mô hình hóa sự phụ thuộc giữa các nút trong biểu đồ cho phép tạo ra bước đột phá trong lĩnh vực nghiên cứu liên quan đến phân tích đồ thị. 
<img src="/images/gnn/3.png">

Bài viết này nhằm mục đích giới thiệu những kiến thức cơ bản về Graph và ứng dụng của chúng.
### What is Graph
Đồ thị là một kiểu dữ liệu phi tuyến tính(non euclidean) có cấu trúc với các nút (còn gọi là đỉnh) và các cạnh. Nó được biểu diễn về mặt toán học là **G(V,E)**. Ví dụ, ***một phân tử NO₂*** có thể được coi là một đồ thị trong đó nguyên tử nitơ và hai nguyên tử oxy được coi là nút, và liên kết giữa các nguyên tử được coi là cạnh. Một ví dụ khác về biểu đồ có thể là gia đình của bạn, trong đó mỗi người là một nút và mối quan hệ giữa hai người là cạnh.
<div class="gallery-box">
<div class="gallery">
<img src="/images/gnn/no2.png">
</div>
</div>
Một đồ thị có thể có nhãn trên cả các nút và các cạnh của nó. Nó có thể là số hoặc văn bản. Mỗi nút sẽ có một số tính năng xác định nút. Trong đồ thị của NO₂, các nguyên tố như số hiệu nguyên tử, khối lượng nguyên tử và số electron hóa trị của mỗi nút có thể là các đặc trưng tương ứng của nó. Các cạnh có thể có hoặc không có các đặc trưng tùy thuộc vào loại biểu đồ. Trong NO₂, các đặc điểm của các cạnh có thể là độ bền của liên kết, loại liên kết (liên kết đơn hoặc liên kết đôi), v.v.

Đồ thị được phân loại trên nhiều cơ sở khác nhau. Phổ biến nhất là dựa trên các cạnh của đồ thị. Đây là những đồ thị có hướng và vô hướng. Trong đồ thị có hướng, các cạnh từ nút này đến nút khác có hướng, trong khi trong đồ thị vô hướng, các nút được nối với nhau qua các cạnh và không có hướng.
<div class="gallery-box">
  <div class="gallery">
    <img src="/images/gnn/1.png">
    <img src="/images/gnn/2.png">
  </div>
</div>


Một ví dụ thực tế về biểu đồ có hướng là Instagram. Khi bạn theo dõi ai đó, họ không nhất thiết phải theo dõi lại bạn. Theo một nghĩa nào đó, điều này là một chiều. Mặt khác, yêu cầu kết bạn trên Facebook là một ví dụ về biểu đồ vô hướng. Sau khi yêu cầu kết bạn được chấp nhận, cả hai bạn có thể xem nội dung của nhau.

### Example Graph
##### Mạng xã hội: 
Mạng xã hội là một biểu đồ trong đó các nút đại diện cho mọi người và mối quan hệ giữa hai người là cạnh. Mối quan hệ này có thể là bất cứ điều gì, từ một người quen đơn giản đến một gia đình.
<img src="/images/gnn/4.png">

#### Phân tử: 
Một phân tử có thể được biểu diễn dưới dạng đồ thị trong đó các nút đại diện cho các nguyên tử và các cạnh đại diện cho liên kết giữa chúng.

<div class="gallery-box">
<div class="gallery">
<img src="/images/gnn/5.png" height=1000px width=400px>
</div>
</div>

#### Internet: 
Internet là một biểu đồ trong đó các thiết bị, bộ định tuyến, trang web và máy chủ là các nút và kết nối internet là các cạnh.
<img src="/images/gnn/6.png">

### Graph explore
#### Phân loại nút - Dự đoán nhãn của một nút nhất định. 
Ví dụ: một người nhất định trong mạng xã hội có thể được phân loại dựa trên sở thích, niềm tin hoặc đặc điểm của họ.
<div class="gallery-box">
<div class="gallery">
<img src="/images/gnn/7.png">
</div>
</div>

#### Dự đoán liên kết 
Dự đoán nếu và cách hai nút được liên kết. Ví dụ, tìm xem hai người (nút) nhất định có bất kỳ mối quan hệ nào giữa họ hay không.
<div class="gallery-box">
<div class="gallery">
<img src="/images/gnn/12
.png">
</div>
</div>

#### Phân cụm - Xác định các cụm nút được liên kết dày đặc. 
Ví dụ: tìm xem một nhóm người có bất kỳ điểm nào giống nhau về chủ đề hay không.

<img src="/images/gnn/8.png">

#### Dự đoán sự tương đồng
Đo mức độ tương tự của hai nút/mạng. Tại đây bạn có thể tìm xem hai người hoặc hai nhóm người khác nhau có giống với nhau hay không.
<div class="gallery-box">
<div class="gallery">
<img src="/images/gnn/11.png">
</div>
</div>

### Graphs Neural Networks Aplication
#### Hệ thống khuyến nghị: 
<img src="/images/gnn/9.png">
Khả năng của hệ thống khuyến nghị có thể được tăng lên theo cấp số nhân bằng cách sử dụng GNN. Với GNN, các đề xuất sẽ dựa trên việc mượn thông tin từ các nút lân cận, do đó làm cho việc nhúng nút chính xác hơn. Pinterest sử dụng hệ thống đề xuất dựa trên GNN.

#### Sự phát triển thuốc: 
<img src="/images/gnn/10.jpg">
Tất cả các phân tử có thể được biểu diễn dưới dạng đồ thị. Sử dụng GNN, có thể lập mô hình các mạng phức tạp như mạng tương tác protein-protein (PPI) và mạng trao đổi chất. Mô hình này giúp phát triển các loại thuốc tốt hơn và ổn định cho bệnh tật.

#### Phân cực trên Twitter: 
<div class="gallery-box">
<div class="gallery">
<img src="/images/gnn/12.jpg">
</div>
</div>
Tùy thuộc vào bài đăng mà một người thích và những người họ theo dõi, có thể phát hiện xem một người có phân cực theo quan điểm cụ thể về một chủ đề (chính trị, môi trường, v.v.) hay không.

#### Phát hiện vòng kết nối xã hội: 
Sử dụng GNN, có thể phát hiện vòng kết nối xã hội của một người dựa trên tương tác của anh ta với những người khác. Vòng kết nối này có thể là đồng nghiệp, bạn đại học, thành viên gia đình, những người cùng lớp, v.v.

### Tại sao không thể áp dụng tích chập cho đồ thị?
Hình ảnh có kích thước cố định và dữ liệu cấu trúc dựa trên lưới với vị trí không gian xác định. Mặt khác, đồ thị có kích thước tùy ý, cấu trúc liên kết phức tạp và cấu trúc phi euclide. Nó cũng không có thứ tự nút cố định. Như chúng ta đã biết, mạng nơ-ron được xác định cho các kích thước, lưới và cấu trúc cụ thể. Do đó không thể áp dụng tích chập trực tiếp cho đồ thị.


