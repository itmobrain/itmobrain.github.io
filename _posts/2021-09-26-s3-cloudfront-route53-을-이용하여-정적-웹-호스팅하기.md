---
layout: post
title: "S3, CloudFront, Route53을 이용하여 정적 웹 호스팅하기"
date: 2021-09-26 16:00:00
image: 'https://user-images.githubusercontent.com/26498433/128895114-720e4bb7-e2af-49d0-9815-1b2ec447d119.png'
description: "원하는 도메인 주소를 가진 정적 웹 사이트를 호스팅하는 방법을 알아봅시다"
tags:
- AWS
- S3
- CloudFront
- Route53
- Web
categories:
- AWS
- S3
- CloudFront
- Route53
- Web
---
## 📌 0. 시작하기 전에
- 해당 글은 S3로 정적 호스팅중인 웹에 CloudFront를 연동하는 과정에서 발생한 에러를 해결(삽질)하기위해 접근한 방법을 기록하였습니다.
- 추가적으로 기본적인 S3와 CloudFront의 연동, 그리고 Route53을 이용한 도메인(그 중에서도 서브도메인) 연결 과정을 간략하게 다룹니다.

***

## 📌 1. 개요(큰 그림 잡기)
<img src="https://user-images.githubusercontent.com/26498433/134797546-f55e6ac2-9a29-444b-bade-10dd261db09f.jpeg"/>
*발 그림 죄송합니다*

<img src="https://user-images.githubusercontent.com/26498433/134797567-12c2b559-4e1e-4572-9cde-fa43c8939ce1.png"/>
*좀 더 정제된 개요도(출처 : [링크](https://thiwankawickramage.medium.com/configure-aws-route-53-cloudfront-and-ssl-certificate-8d64b277d5b3))*

### (1) Route 53
- AWS가 제공하는 DNS 서비스입니다.
- 우리가 S3로 웹을 호스팅하게되면 AWS가 생성해주는 URL을 사용하게 되는데, 이 URL을 우리가 가진 주소(예. www.zzanzu.co.kr)와 연결해주는 역할을 하게 됩니다. 사용자들은 Route53에 적용한 URL로 웹페이지에 접속하게 됩니다.

### (2) CloudFront
- AWS가 제공하는 CDN 서비스입니다.
- S3로 호스팅한 웹에 HTTPS를 적용하고 AWS 인프라의 Edge location(전세계)에  빠르게 전송할 수 있게 해주는 역할을 합니다.

### (3) S3(Static web hosting)
- AWS가 제공하는 스토리지 서비스입니다.
- 보통은 DB에 저장하기엔 용량이 큰 파일들이나 로그, 텍스트 파일 등을 저장하기위해 사용하지만(용도가 다양합니다), 우리는 코드를 업로드하여 S3가 제공해주는 URL을 통해 접속할 수 있는 웹 페이지를 생성하기위해 사용합니다.

***

## 📌 2. S3 정적 웹 호스팅

(1) S3 콘솔에 접속하여 '버킷 만들기' 버튼을 클릭합니다.

(2) 버킷 이름을 설정한 뒤(보유한 도메인을 연결할 것이라면 어떤 이름도 상관없습니다), '모든 퍼블릭 엑세스 차단'에 체크된 설정을 해제한 뒤 '버킷 만들기'를 클릭합니다.

<img src="https://user-images.githubusercontent.com/26498433/134797609-a5237d57-80c7-4419-83b6-b2b8839c4549.png"/>

(3) 생성된 버킷에 간단한 웹페이지 코드(여기서는 index.html)를 업로드하도록 하겠습니다. 여러분들께서는 이후에 호스팅하고자 하는 웹 코드를 업로드하시면 됩니다.

```html
<!--index.html-->
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <title>테스트 페이지</title>
  </head>
  <body>
    <p>테스트 페이지입니다.</p>
  </body>
</html>
```

<img src="https://user-images.githubusercontent.com/26498433/134797626-ca1a045d-3050-4e85-8e99-9867f6b9a8d3.png"/>
*파일을 드래그하여 업로드할 수 있습니다.*

<img src="https://user-images.githubusercontent.com/26498433/134797642-1d2e237b-936c-4bb4-8fcd-e9d2bc76a589.png"/>
*업로드 완료.*

(4) 버킷의 '속성' 탭의 가장 아래에 있는 '정적 웹 사이트 호스팅'의 '편집' 버튼을 누릅니다.
<img src="https://user-images.githubusercontent.com/26498433/134797662-5b423d8e-38fd-487f-91b9-3d686223fe00.png"/>

(5) '정적 웹 사이트 호스팅'을 활성화하고, '인덱스 문서'에 이전에 업로드한 index.html을 입력합니다.
<img src="https://user-images.githubusercontent.com/26498433/134797684-2386b893-5508-419c-a826-5a467a30727a.png"/>

(6) 버킷의 '권한' 탭의 '버킷 정책'에 아래의 스크립트를 추가합니다. Resource 필드의 {버킷 이름}에는 (2)에서 설정한 버킷의 이름을 입력합니다.

```html
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": "*",
            "Action": [
                "s3:GetObject"
            ],
            "Resource": "arn:aws:s3:::{버킷 이름}/*"
        }
    ]
}
```

<img src="https://user-images.githubusercontent.com/26498433/134797722-25c39f7d-bdb1-409b-ad53-7ae7fc8cf43d.png"/>

(7) 다시 '속성' 탭으로 돌아가서 가장 아래 '정적 웹 사이트 호스팅'의 '버킷 웹 사이트 엔드포인트'에 표시된 URL에 접속하면 (3)에서 만들었던 index.html이 표시되는 것을 확인할 수 있습니다.
<img src="https://user-images.githubusercontent.com/26498433/134797739-82429ddc-9846-4f8e-b6fa-2e930a77e2d0.png"/>
<img src="https://user-images.githubusercontent.com/26498433/134797758-a8549f31-dc1e-44f4-9a0f-2dc76057c9f2.png"/>

### 정리해볼까요?
- 간단한 웹 코드를 S3 버킷에 업로드하여 URL을 통해 웹페이지를 생성하게 되었습니다.
- 웹페이지 URL은 다음과 같습니다.

    `http://{버킷 이름}.s3-website.{버킷 region}.amazonaws.com`

***

## 📌 3. S3와 CloudFront 연동
(1) CloudFront 콘솔에 접속하여 '배포 생성'을 클릭합니다.

(2) 여기서 중요한데요, 원본 도메인의 드롭다운을 클릭하게되면 여러 S3 버킷들의 목록이 보일 겁니다.
<img src="https://user-images.githubusercontent.com/26498433/134797915-800c90fe-a57f-48c5-8ff6-6233bfed7ee7.png"/>
*버킷을 선택하게되면 다음과 같이 도메인 이름이 입력된 것을 볼 수 있습니다.*

<img src="https://user-images.githubusercontent.com/26498433/134797933-4706ef8c-28c0-464a-85cd-3cadbe845e1b.png"/>

```html
{버킷 이름}.s3.{버킷 Region}.amazonaws.com
```

여기서 이 도메인 이름을 다음과 같이 수정합니다.(**`s3` → `s3-website`**)

```html
{버킷 이름}.s3-website.{버킷 Region}.amazonaws.com
```

<img src="https://user-images.githubusercontent.com/26498433/134797953-f3c1b6ab-d129-4bfe-9003-d9eed7de5d7a.png"/>

(3) 저희는 웹페이지를 HTTPS만 사용하기 위해 '뷰어 프로토콜 정책'에서 'Redirect HTTP to HTTPS'를 선택하도록 하겠습니다. 

(4) 완료하였으면 '배포 생성'을 클릭합니다. 대략 3분 정도 후에 배포가 완료될 때까지 기다립니다.
<img src="https://user-images.githubusercontent.com/26498433/134797961-7bb4d6bf-2243-4d03-a5b4-8abb2b7d663c.png"/>

(5) '원본' 탭에서 '원본 도메인'에 (2)에서 입력한 URL 형식이 제대로 되었는지 다시 한 번 확인합니다.

```html
{버킷 이름}.s3-website.{버킷 Region}.amazonaws.com
```
<img src="https://user-images.githubusercontent.com/26498433/134797981-4c68626c-5354-4067-b81e-a511b05caf11.png"/>

(6) '세부 정보'의 '배포 도메인 이름'의 URL(~~.cloudfront.net)을 복사한 뒤, 웹페이지에서 열어보도록 하겠습니다. 정상적으로 표시되는 것을 확인할 수 있습니다.
<img src="https://user-images.githubusercontent.com/26498433/134798044-f60a0bb4-329f-47df-b323-c02b3cf7aabc.png"/>

### 정리해볼까요?
- S3 웹 호스팅 버킷을 CloudFront와 연동하였습니다.
- 웹페이지 URL은 다음과 같습니다.
    `https://@@@.cloudfront.net`

***

## 📌 4. CloudFront에 Route53 도메인 연결

### 4.1 도메인 구매
- 기존에 이미 구입한 도메인이 있는 경우에는 이 과정은 스킵해도 되고, 그렇지 않다면 별도로 도메인을 구매하거나 AWS의 Route53을 통해 구매할 수 있습니다([가이드](https://docs.aws.amazon.com/ko_kr/Route53/latest/DeveloperGuide/domain-register.html)).

### 4.2 SSL 인증서 발급
- AWS Certificate Manager(ACM)를 통해 위에서(`4.1`) 구매한 도메인을 입력합니다.
- 저같은 경우는 `zzanzu.@@@@.tv` 처럼 서브 도메인을 이용할 것이기 때문에 `*.@@@@.tv` 형식으로도 인증서를 발급합니다.
- 다음의 링크에서 발급하는 과정을 참고하시면 좋을 것 같습니다. [참고](https://jojoldu.tistory.com/434)

### 4.3 CloudFront에 도메인 연결
- 위에서 생성한 CloudFront의 생성한 배포(distribution)를 클릭하여 '일반' 탭의 '설정'에서 '편집' 버튼을 클릭합니다.
    <img src="https://user-images.githubusercontent.com/26498433/134798117-b8c01501-62fd-4af9-b100-d6c99a2419ce.png"/>

- '설정' 항목의 '대체 도메인 이름(CNAME)'에 부여하고싶은 주소를 입력합니다.
    <img src="https://user-images.githubusercontent.com/26498433/134798133-0cf54903-6627-4a48-8e39-96753ce2690c.png"/>

- '사용자 정의 SSL 인증서'에서 `4.1`에서 발급받은 인증서를 선택하고, '변경 사항 저장'을 클릭하여 배포되기를 기다립니다(시간이 조금 걸리니 상심하지 마시고 기다려주세요)
    <img src="https://user-images.githubusercontent.com/26498433/134798162-b9361db0-786c-4215-a69c-7beadb0df141.png"/>

- Route53 콘솔로 이동하여 호스팅 영역 내의 연결할 도메인을 클릭합니다.
    <img src="https://user-images.githubusercontent.com/26498433/134798180-0e2531f6-d982-4267-9bd1-e65f5701e85c.png"/>

- 연결할 도메인을 클릭한 뒤, '레코드 생성'을 클릭합니다.
    <img src="https://user-images.githubusercontent.com/26498433/134798210-9f744c31-6a47-4f8f-9212-d9be3035700e.png"/>
    <img src="https://user-images.githubusercontent.com/26498433/134798232-7a3a4984-9c45-4f54-80dd-0702d070f426.png"/>

- '레코드 이름'에 연결할 도메인 주소를 입력하고, '레코드 유형'은 A, '별칭' 스위치를 활성화 한 후 '트래픽 라우팅 대상'은 'CloudFront 배포에 대한 별칭' 선택 후 미리 생성한 CloudFront 배포의 URL(`@@@@.cloudfront.net`)을 입력한 뒤 '레코드 생성'을 클릭합니다.
    <img src="https://user-images.githubusercontent.com/26498433/134798242-de993cb6-8bf5-4dee-b9ae-746a5ea80475.png"/>

- 잠시 기다린 뒤 생성한 도메인 주소를 입력하면 정상적으로 웹페이지가 등록된 것을 확인할 수 있습니다. 수고하셨습니다.
    
    <img src="https://user-images.githubusercontent.com/26498433/134798257-15d5a93e-8728-4b0f-84eb-a4113d29c78a.png"/>
