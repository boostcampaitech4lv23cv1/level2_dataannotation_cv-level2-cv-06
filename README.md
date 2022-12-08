# Rule
> Rule : BlackFormatter로 설정
> - 클래스명 : PascalCase
> - 함수명, 변수명 : snake_case
> - 각 함수, 클래스 사이 두 줄 띄우기
> - annotation(type hint) 사용
> - 간단한 docstring 작성
> - 기능 만들 때 feat/기능 내용으로 브랜치 명 작성
> - 데이터 분석시 EDA/분석 내용으로 브랜치 명 작성
> 
> ex)
> ```py
> "feat/gethyperparams"
> "EDA/rgb_analysis"
> ```

> Commit Convention
> - feat, fix, docs, refactor, test, remove, add
> - 세부 내용 적기
> (fix는 오류or버그 고쳤을 때, refactor는 코드 수정했을 때)
> 
> ex)
>  ```bash
>  "feat: get model hyper parameter"
>  "세부 내용 추가"
>  ```

# Dataset information
> [ICDAR 2017](https://rrc.cvc.uab.es/?ch=8)
> - 로그인 후 Multi-script text detection 데이터 다운로드
> - image 파일은 input/data/ICDAR17/images 폴더에 저장
> - ufo 파일은 input/data/ICDAR17/ufo 폴더에 저장
