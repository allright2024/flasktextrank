from flask import Flask, render_template, request
import google_keyword
import textrank #새로 추가한 것

#Flask 객체 인스턴스 생성
app = Flask(__name__)

@app.route('/',methods=('GET', 'POST')) # 접속하는 url
def index():
    # 웹 페이지에서 name="xxx"인 요소의 value 가져오기
    print(request.form.get('keyword1'))
    keyword1 = request.form.get('keyword1')

    # 위의 값이 있을 때만 크롤링 검색 결과 반환
    '''if keyword1 is not None:
        data = {
            keyword1 : google_keyword.get_search_count(keyword1).get('number'),
            
            }
        return render_template('index.html',data=data)
    else:
        return render_template('index.html')'''
    
    if keyword1 is not None:
        data = {
            keyword1 : textrank.newssum(keyword1),
        }
        return render_template('index.html', data=data)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
