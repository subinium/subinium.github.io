---
title: "데이터셋을 위한 데이터시트(Datasheets for Datasets)"
category:
  - ai
tag:
  - datasets
  - motivation
  - composition
  - collection process
  - uses
  - distribution
  - maintenance
  - 데이터셋
  - 매뉴얼
author_profile: true
header:
  overlay_image: https://images.unsplash.com/photo-1529078155058-5d716f45d604?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1649&q=80
  overlay_filter: 0.7
---
데이터셋을 만들 때, 투명성과 책임을 위해 고려할 점들

- 다음 질문들에 대해서 존재 여부를 묻는 질문에 대해서 존재하는 경우 설명을 추가해주시는게 좋습니다.
  - 설명의 경우, 단순한 예/아니오를 넘어 이유, 웹사이트 링크, 방법론, 절차 등을 모두 포함합니다.
  - 질문에 대한 구체적인 답변은 논문을 참고하면 더 좋습니다.

데이터셋의 투명성과 책임을 위해 노력합시다.

- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)

## 1. 동기(Motivation)

- **데이터셋은 어떤 목적을 위해 만들어졌습니까?** 
  - *For What purpose was the dataset created*
- **데이터셋을 누가(e.g, 팀, 연구집단) 어떤 엔티티(예 : 회사, 학회, 협회 등)를 대신하여 만드었습니까?** 
  - *Who created the dataset (예 : which team, research group) and on behalf of which entity (예 : company, institution, organization)?*
- **누가 데이터셋 생성에 (금전적) 지원을 했습니까?** 
  - *Who funded the creation of the dataset?*

## 2. 구성(Composition)



- **데이터셋을 구성하는 인스턴스는 무엇입니까(예 : 문서, 사진, 사람, 국가)?** 
  - *What do the instances that comprise the dataset represent (예 : documents, photos, people, countries)?*
- **총 몇 개의 인스턴스가 있습니까?** 
  - *How many instances are there in total (of each type, if appropriate)?*
- **데이터셋은 모든 인스턴스를 포함하고 있습니까? 아니면 큰 셋의 샘플 데이터셋입니까?**
  - *Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?*
- **각 인스턴스는 어떤 데이터로 구성됩니까? raw 데이터 또는 feature?** 
  - *What data does each instance consist of? “Raw” data (예 : unprocessed text or images) or features?* 
- **각 인스턴스와 관련된 레이블 또는 타겟값이 있습니까?** 
  - *Is there a label or target associated with each instance?*
- **개별 인스턴스에서 누락된 정보가 있습니까?**
  - *Is any information missing from individual instances?* 
- **개별 인스턴스 간의 관계가 명시되어 있습니까 (예 : 사용자의 영화 등급, 소셜 네트워크 링크)?**
  - *Are relationships between individual instances made explicit (예 : users’ movie ratings, social network links)?*
- **권장 데이터 분할(training, development/validation, testing)이 있습니까?**
  - *Are there recommended data splits (예 : training, development/validation, testing)?*
- **데이터셋에 오류, 노이즈 또는 중복이 있습니까?**
  - *Are there any errors, sources of noise, or redundancies in the dataset?* 
- **데이터셋가 자체 포함되어 있습니까, 아니면 외부 리소스 (예 : 웹 사이트, 트윗, 기타 데이터셋)에 연결되거나 의존합니까?**
  - *Is the dataset self-contained, or does it link to or otherwise rely on external resources (예 : websites, tweets, other datasets)?* 
- **데이터셋에 기밀로 간주될 수있는 데이터가 포함되어 있습니까 (예 : 법적 권한 또는 의사 환자 기밀로 보호되는 데이터, 개인의 비공개 커뮤니케이션 내용이 포함 된 데이터)?**
  - *Does the dataset contain data that might be considered confidential (예 : data that is protected by legal privilege or by doctorpatient confidentiality, data that includes the content of individuals’ non-public communications)?* 
- **데이터셋에 직접 볼 경우 공격적이거나 모욕적이거나 위협적이거나 불안을 유발할 수있는 데이터가 포함되어 있습니까?**
  - *Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?*
- <b style="color:blue">데이터셋이 사람과 관련이 있습니까? 그렇지 않은 경우이 섹션의 나머지 질문을 건너 뛸 수 있습니다.</b>
  - *Does the dataset relate to people? If not, you may skip the remaining questions in this section.*
- **데이터셋가 하위 집단을 식별합니까 (예 : 연령, 성별)?**
  - *Does the dataset identify any subpopulations (예 : by age, gender)?*
- **데이터셋에서 개인(한 명 이상의 일반인)을 직접 또는 간접적으로 (다른 데이터와 결합하여) 식별 할 수 있습니까?**
  - *Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?*
- **데이터셋에 어떤 방식으로든 민감한 것으로 간주될 수있는 데이터가 포함되어 있습니까 (예 : 인종 또는 민족 출신, 성적 지향, 종교적 신념, 정치적 견해 또는 조합원, 위치, 재무 또는 건강 데이터, 생체 인식 또는 유전 데이터; 사회 보장 번호와 같은 정부 신분증, 범죄 기록)?**
  - *Does the dataset contain data that might be considered sensitive in any way (예 : data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?*


## 수집 과정(Collection Process)

- **각 인스턴스와 관련된 데이터는 어떻게 수집 되었습니까? 데이터를 직접 관찰 할 수 있었습니까 (예 : raw 텍스트, 영화 등급), 피험자가보고했거나 (예 : 설문 조사 응답), 또는 다른 데이터에서 간접적으로 추론/파생 했습니까 (예 : 품사 태그, 연령에 대한 모델 기반 추측 또는 언어)?**
  - *How was the data associated with each instance acquired? Was the data directly observable (예 : raw text, movie ratings), reported by subjects (예 : survey responses), or indirectly inferred/derived from other data (예 : part-of-speech tags, model-based guesses for age or language)?*
- **데이터가 피험자에 의해 보고되거나 다른 데이터에서 간접적으로 추론/유추 된 경우, 데이터가 검증되었습니까?**
  - *If data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified?*
- **데이터 수집에 사용 된 메커니즘 또는 절차 (예 : 하드웨어 장치 또는 센서, 수동 인간 큐 레이션, 소프트웨어 프로그램, 소프트웨어 API)는 무엇입니까? 이러한 메커니즘 또는 절차는 어떻게 검증 되었습니까?**
  - *What mechanisms or procedures were used to collect the data (예 : hardware apparatus or sensor, manual human curation, software program, software API)? How were these mechanisms or procedures validated?*
- **데이터셋이 더 큰 집합의 샘플인 경우 샘플링 전략은 무엇이었습니까 (예 : 결정론적, 확률적)?**
  - *If the dataset is a sample from a larger set, what was the sampling strategy (예 : deterministic, probabilistic with specific sampling probabilities)?*
- **데이터 수집 프로세스에 참여한 사람 (예 : 학생, 크라우드 워커, 계약자)은 누구이고, 어떻게 보상받았습니까? (예 : 크라우드 워커에게 지급 된 금액)**  
  - *Who was involved in the data collection process (예 : students, crowdworkers, contractors) and how were they compensated (예 :how much were crowdworkers paid)?*
- **데이터가 수집된 기간은 얼마입니까? 이 기간이 인스턴스와 관련된 데이터의 생성 기간과 일치합니까 (예 : 오래된 뉴스 기사의 최근 크롤링)?**
  - *Over what timeframe was the data collected? Does this timeframe match the creation timeframe of the data associated with the instances (예 : recent crawl of old news articles)?*
- **윤리적 리뷰 프로세스가 수행되었습니까?**
  - *Were any ethical review processes conducted (예 : by an institutional review board)?* 
- <b style="color:blue">데이터셋이 사람과 관련이 있습니까? 그렇지 않은 경우이 섹션의 나머지 질문을 건너 뛸 수 있습니다.</b>
  - *Does the dataset relate to people? If not, you may skip the remaining questions in this section.*
-  **개인으로부터 데이터를 직접 수집 했습니까, 아니면 제 3자 또는 다른 출처 (예 : 웹 사이트)를 통해 수집 했습니까?**
  - *Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (예 : websites)?*
- **해당 개인은 데이터 수집에 대한 알림을 받았습니까?**
  - *Were the individuals in question notified about the data collection?*
- **해당 개인이 데이터 수집 및 사용에 동의했습니까?**
  - *Did the individuals in question consent to the collection and use of their data?*
- **동의를 얻은 경우, 동의 한 개인에게 향후 또는 특정 용도로 동의를 취소할 수 있는 메커니즘이 제공 되었습니까?**
  - *If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?*
- **데이터셋의 잠재 영향과 데이터 주체에 대한 사용에 대한 분석 (예 : 데이터 보호 영향 분석)이 수행 되었습니까?**
  - *Has an analysis of the potential impact of the dataset and its use on data subjects (예 : a data protection impact analysis)been conducted?*

## 전처리/클리닝/라벨링(Preprocessing/cleaning/labeling)

- <b style="color:blue"> 데이터의 전처리/클리닝/라벨링(이하 처리)이 수행 되었습니까? 그렇지 않은 경우이 섹션의 나머지 질문을 건너 뛸 수 있습니다.</b>
  - *Was any preprocessing/cleaning/labeling of the data done (예 :discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? If not, you may skip the remainder of the questions in this section.*
- **위의 처리를 제외한 'raw' 데이터가 저장되었습니까?**
  - *Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (예 : to support unanticipated future uses)?* 
- **인스턴스 처리하는데 사용한 소프트웨어를 사용할 수 있습니까?**
  - *Is the software used to preprocess/clean/label the instances available?* 

## 사용(Uses)

- **이미 이 데이터셋을 사용한 task가 있습니까?**
  - *Has the dataset been used for any tasks already?*
- **데이터셋을 사용하는 일부 또는 모든 문서 또는 시스템에 연결되는 저장소가 있습니까?**
  - *Is there a repository that links to any or all papers or systems that use the dataset?*
- **데이터셋은 어떤 (기타) 작업에 사용될 수 있습니까?**
  - *What (other) tasks could the dataset be used for?*
- **향후 사용에 영향을 미칠 수있는 데이터셋의 구성 또는 수집 및 처리 방식에 관한 사항이 있습니까?**
  - *Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?*
- **데이터셋을 사용해서는 안되는 작업이 있습니까?**
  - *Are there tasks for which the dataset should not be used?*

## 배포(Distribution)

- **데이터셋을 생성한 엔티티 외의 외부의 제3자에게 데이터셋가 배포됩니까?**
  - *Will the dataset be distributed to third parties outside of the entity (예 : company, institution, organization) on behalf of which the dataset was created?*
- **데이터셋은 어떻게 배포되나요 (예 : API, GitHub)? 데이터셋에 디지털 개체 식별자 (DOI)가 있습니까?**
  - *How will the dataset will be distributed (예 : tarball on website, API, GitHub)? Does the dataset have a digital object identifier (DOI)?*
- **데이터셋은 언제 배포됩니까?**
  - *When will the dataset be distributed?*
- **데이터셋은 저작권 또는 기타 지적 재산권 (IP) 라이선스 또는 해당 사용 약관 (ToU)에 따라 배포됩니까?**
  - *Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?*
- **제3자가 인스턴스와 관련된 데이터에 IP 기반 또는 기타 제한을 부과했습니까?**
  - *Have any third parties imposed IP-based or other restrictions on the data associated with the instances?* 
- **데이터셋 또는 개별 인스턴스는 수출 통제 또는 기타 규제 제한이 적용됩니까?**
  - *Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?* 

## 유지보수(Maintenance)

- **누가 데이터셋를 지원/호스팅/유지하고 있습니까?**
  - *Who is supporting/hosting/maintaining the dataset?*
- **데이터셋 소유자/큐레이터/관리자에게 연락하는 방법 (예 : 이메일 주소)?**
  - *How can the owner/curator/manager of the dataset be contacted (예 : email address)?*
- **정오표가 있습니까?**
  - *Is there an erratum?*
- **데이터셋은 업데이트 됩니까? (예 : 라벨 지정 오류 수정, 새 인스턴스 추가, 인스턴스 삭제)?**
  - *Will the dataset be updated (예 : to correct labeling errors, add new instances, delete instances)?*
- **데이터셋이 사람과 관련된 경우 인스턴스와 관련된 데이터 보존에 적용 가능한 제한이 있습니까 (예 : 문제의 개인이 자신의 데이터가 고정 된 기간 동안 보존 된 다음 삭제된다고 말했습니까)?**
  - *If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (예 : were individuals in question told that their data would be retained for a fixed period of time and then deleted)?*
- **이전 버전의 데이터셋이 계속 지원/호스팅/유지됩니까?**
  - *Will older versions of the dataset continue to be supported/hosted/maintained?*
- **다른 사람들이 데이터셋에 확장/증가/구축/기여하고 싶다면 그렇게 할 수있는 메커니즘이 있습니까?**
  - *If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?*