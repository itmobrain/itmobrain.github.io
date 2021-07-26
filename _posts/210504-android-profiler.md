---
layout: post
title: "메모리 사용량이 많은 코드를 찾는 방법"
date: 2021-05-04 15:34:35
image: 'https://user-images.githubusercontent.com/26498433/117010167-7d277600-ad27-11eb-816c-a723afb4aad6.png'
description: profiler 기능에 대하여.
tags:
- android
categories:
- android
- profiler
---
## 1. 개요 📌
- 카메라 입력 이미지를 사용하는 로직을 구현하다보면 연산량이 많아서 그런지 현재 기준 상대적으로 저사양의 기종(ex. 갤럭시 노트3)들에서 ANR이 많이 발생한다.
- 로직상의 에러는 아닌데, 유난히 버벅이는 기기들에서의 상황을 해결하려다보니 좀 더 딥하게 원인을 분석할 필요가 생겼다.
- GC 관련 로그가 찍히는 것을 보고 막연하게 메모리 에러라는 가정을 하고 해결해볼 방법을 생각해봤다.
- 이럴 땐 Android Studio에서 제공하는 profiler 기능을 활용하면 된다.

### 예시

<img src="https://user-images.githubusercontent.com/26498433/117009935-3c2f6180-ad27-11eb-9da9-c053011f72d3.png">


## 2. 활용 방법 📌
- Memory 영역 클릭 > 메모리 에러가 발생하는 것 같은 상황을 Record > 메모리가 많이 할당된 변수 타입 선택 > 오른쪽 클릭 후 'Jump to Source' 선택

<img src="https://user-images.githubusercontent.com/26498433/117010167-7d277600-ad27-11eb-816c-a723afb4aad6.png">

- 어떤 부분에서 메모리가 많이 사용, 할당되었는지 원인을 분석하는데에 실마리를 얻을 수 있다.
