# SDH 데이터수집  

## 개선점   

기존의 수집법에서 두가지 부분에 중점을 두고 개선을 했습니다.  

- 가능한한 IO작업을 줄일것    
- 프로세스 끼리 병렬작업시 공유하는 내용이 없을것

실제  2011/05/22 부터 2012/12/31 까지 Sutardja Dai Hall 데이터를 받는데 약 22시간에서 6시간정도로 수집 시간이 줄어들었습니다. VTT데이터 외에도 추가로 데이터를 수집하고자 할때 도움이 되셨으면 좋겠습니다.

## 구동설명  

### const.py  
const.py에 보시면 데이터를 수집하는 데상에 따라서 설정을 변경해줘야 할 변수들이 있습니다. 

1. META_URL : 메타데이터를 수집할수 있는 URL입니다.  
2. DATA_URL : 실제 데이터를 수집하는  URL입니다. 
3. SDH_META_FOLDER : 메타데이터를 저장하는곳
4. SDH_BIN_FOLDER : BIN파일을 저장하는곳


### retrieve_sdh.py  
retrieve_sdh.py에 보시면 맨 아랫쪽에 다음과 같은 라인을 수정하여 데이터 수집기간을 설정합니다.

    start_time = dt.time(hour = 0, minute = 0, second = 0)
    start_date = dt.datetime.combine(dt.date(2011, 5, 22), start_time)
    end_date = dt.datetime.combine(dt.date(2012, 12, 31), start_time)

모든 설정이 완료되시면 아래와 같이 수집코드를 작동하여 데이터를 수집합니다.

    python retrieve_sdh.py
