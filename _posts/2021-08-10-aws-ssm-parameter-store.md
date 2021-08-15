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
## 📌 1. 읽기 전에
> <cite>본 글에서는 AWS의 System Manager(구 SSM) 전체를 다루는 것이 아닌, 기능의 일부인 Parameter Store에 대해 설명합니다.</cite>

- 코드를 작성하다보면 github 레파지토리에 올라가기에는 민감한 정보들(주로 config 값들)을 어떻게 관리해야할까, 한 번쯤은 고민해보셨을 겁니다. 
  
- 저 또한 이러한 상황에 익숙하지 않은 시절에는 이러한 정보들을 별도의 파일에 정의한 뒤 .gitignore 파일에 추가하여 레포지터리에는 업로드되지 않도록 하던 시절도 있었습니다 -_-;;

- 하지만 이러한 경우 협업 과정에서 공유하는 과정이 번거롭기도 하고, 그렇다고 안전하게 관리되는 느낌이 있지도 않았습니다.

- 그러던 중, AWS의 **Parameter Store**를 통해 이러한 상황을 해결해줄 수 있다는 것을 알게 되었습니다.

- 이 글을 통해 AWS Parameter Store가 무엇이고, 어떤 기능을 제공하는지 이해가 되신다면 좋겠습니다 :)

***

## 📌 2. Parameter Store란?
- AWS System Manager의 여러 기능 중 하나입니다.

- 참고로 System Manager에는 5개 유형의 기능을 제공합니다.
  - Operations Management
  - Application Management
  - Change Management
  - Node Management
  - Shared Resources


- 이 중에서 Parameter Store는 Application Management에 속하는 기능으로서,
    - 우리가 작성한 **코드에서 secret값이나 config값을 분리**시켜주고(암호화도 원한다면 사용 가능)
    - **AWS의 여러 서비스들 내에서 이 값들을 사용할 수 있게** 해줍니다.


### 2.1 parameter가 뭔데?
- Parameter Store의 parameter는 Parameter Store에 저장된 **텍스트 블록이나 이름의 리스트, 비밀번호, 또는 AMI ID와 같은 데이터**를 의미합니다. 이를 통해 우리의 스크립트나 커맨드같은 곳에서 안전히 관리된 채로 사용할 수 있습니다.

- parameter에는 3가지 타입이 있습니다.
    - String
    - StringList
    - SecureString
  
- String과 StringList는 말 그대로 문자열과 문자열 리스트 타입입니다. 말 그대로 문자열과 문자열 리스트를 Parameter Store에 저장해놓고 코드에서 불러오는 형식으로 사용할 수 있는 것이죠.

```
// String 타입 예시
abc123
Example Corp
<img src="images/bannerImage1.png"/>

// StringList 타입 예시 - 쉼표(comma)로 구분된 문자열
Monday,Wednesday,Friday
CSV,TSV,CLF,ELF,JSON
```

> <cite>코드 내에서 뿐만 아니라, 운영중인 서비스(웹, 앱 등) 내에서 공통적으로 사용되는 값(예. 이미지 URL)들은 Parameter Store로 관리한다면 편리할 것 같네요!</cite>

#### SecureString 타입
- 조심스럽게 참조/저장되어야하는 민감한 데이터인 경우 사용되는 타입입니다. 비밀번호나 라이센스 키와같이 사용자들에 의해 바뀌거나 참조되지 않아야하는 데이터가 있는 경우, 이 타입을 사용하여 parameter를 생성하면 됩니다.

- 다시 한 번 강조하지만, 민감한 정보인 경우 String이나 StringList를 쓰면 안됩니다!(Don't) 민감한 정보는 반드시 암호화되어야 하기 때문에, SecureString을 사용해야합니다.

- '그래도 그냥 String으로 사용해도 되지 않을까?' 할 수 있겠지만, 우리가 모르는 사이에 CloudTrail 로그나 일반 커맨드 메세지나 agent 로그에 포함되어있을 수도 있습니다.

- SecureString은 AWS의 KMS key를 이용하여 decrypt, encrypt됩니다. KMS key는 AWS가 기본으로 제공하는 값을 사용해도 되고, 우리가 만든 KMS key를 사용해도 됩니다.

- 다양한 AWS 내의 서비스에서 사용될 수 있는데요, 심지어는 람다 코드 내에서도 parameter 값을 불러와서 사용할 수 있습니다.

```python
from __future__ import print_function
 
import json
import boto3
ssm = boto3.client('ssm', 'ap-northeast-2')

def get_parameters():
    # LambdaSecureString라고 정의한 값을 불러옵니다.
    response = ssm.get_parameters(
        Names=['LambdaSecureString'],WithDecryption=True
    )
    for parameter in response['Parameters']:
        return parameter['Value']
        
def lambda_handler(event, context):
    value = get_parameters()
    print("value1 = " + value)
    return value  # Echo back the first key value
```

***

## 📌 3. 사용해보기
> <cite>python 코드 내에서 SecureString 타입으로 정의한 값을 사용하는 예시입니다.</cite>

### Step 1. parameter 생성 하기
- parameter는 AWS 의 콘솔(웹)에서도 생성이 가능하고, AWS CLI를 통해 생성하는 것도 가능합니다. 아래 예시에서는 콘솔을 통해 생성하는 것으로 설명합니다.

- AWS의 System Manager > 좌측 메뉴에서 Parameter Store 클릭 > Create parameter 클릭 > Name에 parameter 이름 입력 > Type 선택 및 Value 입력 

<img src="https://user-images.githubusercontent.com/26498433/128894733-879fc535-bf62-4849-ba21-151848a98023.png" />

  - **Name**
    - parameter의 이름(변수명이라고 생각하면 편할 것 같습니다)을 선언합니다.
    - 운영중인 여러 서비스 내에서 사용될 수 있기 때문에 하이픈('-')을 이용해 의미를 설명할 수 있겠죠?(갑자기 질문)
    - (예) server-dev-db-url : 개발 서버의 DB URL 정보를 저장합니다.
    
  - **Tier** - [ref](!https://docs.aws.amazon.com/systems-manager/latest/userguide/parameter-store-advanced-parameters.html)
    - Standard : default Tier로써 해당 Region에 최대 10,000개의 parameter를 저장할 수 있게 됩니다. 저의 경우처럼 고작해야 서비스 내의 변수들만 관리하는 용도라면 문제가 없겠지만, 실시간으로 parameter를 생성해야하는 로직이 있다면 Advanced Tier를 고려하는 것이 좋습니다.
    - Advanced : parameter를 100,000개까지 생성할 수 있습니다. 자세한 내용은 ref의 링크를 참고해주세요.
  
  - **Type** : 위에서 설명한 것 처럼 사용하려는 용도에 맞는 타입을 선택합니다.
    
  - **Value** : 실제로 불러올 값을 입력합니다(config, secret 값 등)
    
  - 생성 완료 예시
  <img src="https://user-images.githubusercontent.com/26498433/129469271-f53df10f-30da-4752-9b74-a52aa0be04f9.png" />

### Step 2. Policy 선언 및 코드 구현(파이썬 예시)
- parameter를 생성하였으니, 코드 내에서 잘 불러와지는지 확인해봅시다.

- 우선, 파이썬 코드 내에서 AWS의 서비스(S3, SQS 등)를 접근하기 위해서는 boto3라는 라이브러리를 사용해야합니다.
```bash
$ pip install boto3
```

- 또한 AWS 서비스에 접근하기 위해서는 코드를 작성하는 환경이 해당 권한을 가지고 있다는 것을 증명해야합니다. 증명 방법에는 여러가지가 있는데, AWS CLI를 설치한 뒤 config에 Access Key와 ID를 직접 등록하는 방식도 있고, 코드 내에 직접 추가하는 방식도 있지만(보안적으로 후자의 방식은 별로 추천하지 않는다), IAM policy를 생성하여 코드를 실행하는 인스턴스에 적용된 권한에 추가합니다.

<img src="https://user-images.githubusercontent.com/26498433/129469705-9ee55db1-7718-4f90-b3a1-c43e0a1f9c19.png" />

  - "Action" : 허용할 action을 추가합니다. 여기서는 parameter를 불러오는 기능만 테스트하기 때문에 GetParameter에 대한 권한을 추가했습니다.
  - "Resource" : policy를 통해 접근 가능한 parameter를 추가합니다. 위의 예시에서는 '*'을 입력하여 모든 parameter를 접근할 수 있도록 하였습니다. 

- 아래의 코드를 통해 parameter를 불러옵니다. [boto3 문서 참고](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ssm.html#SSM.Client.get_parameter)

```python
def __get_parameter(parameter_name):
    """
    AWS SSM Parameter Store에 등록된 {parameter_name} 값을 불러옴.
    :return: Parameter Store에 저장된 값
    
    세부적인 api는 boto3 공식 문서 참고
    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ssm.html#SSM.Client.get_parameter
    """
    ssm = boto3.client('ssm', 'ap-northeast-2') # Region 입력
    
    response = ssm.get_parameter(
        Name=parameter_name, # 미리 생성한 parameter의 name값
        WithDecryption=True
    )

    return response['Parameter']['Value']
```
  

[참고](!https://docs.aws.amazon.com/systems-manager/latest/userguide/param-create-cli.html#param-create-cli-securestring)