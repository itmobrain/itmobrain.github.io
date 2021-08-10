---
layout: post
title: "AWS Parameter Store 적용하기"
date: 2021-08-10 23:09:35
image: 'https://user-images.githubusercontent.com/26498433/128895114-720e4bb7-e2af-49d0-9815-1b2ec447d119.png'
description:
tags:
- aws
- python
categories:
- aws
---

## 1. 읽기 전에
> 본 글에서는 AWS의 System Manager(구 SSM) 전체를 다루는 것이 아닌, 기능의 일부인 Parameter Store에 대해 설명합니다.

- 코드를 작성하다보면 github 레포지터리에 올라가기에는 민감한 정보들(주로 config 값들)을 어떻게 관리해야할까, 한 번쯤은 고민해보셨을 겁니다. 이러한 상황에 익숙하지 않은 시절에는 별도의 파일을 만든 뒤 .gitignore 파일에 추가하여 레포지터리에는 업로드되지 않도록 하던 시절도 있었습니다 -_-;;

- 하지만 이러한 경우 협업 과정에서 공유하는 과정이 번거롭기도 하고, 그렇다고 안전하게 관리되는 느낌이 있지도 않았습니다.

- 그러던 중, AWS의 Parameter Store를 통해 이러한 상황을 해결해줄 수 있다는 것을 알게 되었습니다.

## 2. Parameter Store란?
- AWS System Manager(이전에는 SSM이라고 불렸다)의 여러 기능 중 하나입니다.

- 참고로 System Managerdpsms 5개 유형의 기능을 제공합니다.
  ⇒ Operations Management, Application Management, Change Management, Node Management, Shared Resources

- 이 중에서 Parameter Store는 Application Managerment에 속하는 기능으로서,
    - 우리가 작성한 코드에서 secret값이나 config값을 분리시켜주고(암호화도 원한다면 사용 가능하다)
    - AWS의 여러 서비스들 내에서 이 값들을 사용할 수 있게 해준다.

### parameter가 뭔데?
- Parameter Store의 parameter는 Parameter Store에 저장된 텍스트 블록이나 이름의 리스트, 비밀번호, 또는 AMI ID와 같은 데이터를 의미한다. 이를 통해 우리의 스크립트나 커맨드같은 것에서 안전하고 관리되기 용이하도록 사용될 수 있다.

- parameter를 참조할 때는 아래와같은 컨벤션을 사용한다

  {{ssm:parameter-name}}

- parameter에는 3가지 타입이 있다.
    - String
    - StringList
    - SecureString

- 조심스럽게 참조되고 저장되어야하는 민감한 데이터인 경우 사용되는 타입이다. 비밀번호나 라이센스 키와같이 사용자들에 의해 바뀌거나 참조되지 않아야하는 데이터가 있는 경우, 이 타입을 사용하여 parameter를 생성하면 된다.

- 다시 한 번 강조하지만, 민감한 정보인 경우 String이나 StringList를 쓰면 안된다!(Don't) 민감한 정보는 반드시 암호화되어야 하기 때문에, SecureString을 사용해야한다.

- Secure String 생성하기 : [참](https://docs.aws.amazon.com/systems-manager/latest/userguide/param-create-cli.html#param-create-cli-securestring)

- '그래도 그냥 String으로 사용해도 되지 않을까?' 할 수 있겠지만, 우리가 모르는 사이에 CloudTrail 로그나 일반 커맨드 메세지나 agent 로그에 포함되어있을 수도 있기 때문이다!

- SecureString은 KMS key를 이용하여 decrypt, encrypt한다.

- 다양한 AWS 내의 서비스에서 사용될 수 있다. 심지어는 람다에서도 사용될 수 있다!


## 3. 사용해보기
> python 코드 내에서의 사용을 예시로 설명합니다.

- AWS의 System Manager > 좌측 메뉴에서 Parameter Store 클릭 > Create parameter 클릭 > Name에 parameter 이름 입력 > Type 선택 및 Value 입력 

<img src="https://user-images.githubusercontent.com/26498433/128894733-879fc535-bf62-4849-ba21-151848a98023.png" />

- 우선, 파이썬 코드 내에서 AWS의 서비스(S3, SQS 등)를 접근하기 위해서는 boto3라는 라이브러리를 사용해야한다.
```bash
$ pip install boto3
```

- 또한 AWS 서비스에 접근하기 위해서는 코드를 작성하는 환경이 해당 권한을 가지고 있다는 것을 증명해야한다. 증명 방법에는 여러가지가 있는데, AWS CLI를 설치한 뒤 config에 Access Key와 ID를 직접 등록하는 방식도 있고, 코드 내에 직접 추가하는 방식도 있다(보안적으로 후자의 방식은 별로 추천하지 않는다).

- (작성중)