#!/usr/bin/env python3
"""
테스트용 PDF 문서 생성 스크립트
직관적으로 확인 가능한 명확한 사실 정보가 담긴 PDF를 생성합니다.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfbase import pdfutils
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import os

def create_test_pdf():
    """테스트용 PDF 문서 생성"""
    
    # 출력 파일 경로
    output_path = "./data/pdfs/misc/테스트용_회사정보.pdf"
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # PDF 문서 생성
    doc = SimpleDocTemplate(output_path, pagesize=A4, topMargin=2*cm)
    
    # 스타일 설정
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=18,
        spaceAfter=30,
        alignment=1  # 중앙 정렬
    )
    
    # 문서 내용
    story = []
    
    # 제목
    story.append(Paragraph("테크노 솔루션즈 회사 정보", title_style))
    story.append(Spacer(1, 20))
    
    # 회사 기본 정보
    story.append(Paragraph("<b>1. 회사 개요</b>", styles['Heading2']))
    story.append(Paragraph("회사명: 테크노 솔루션즈 주식회사", styles['Normal']))
    story.append(Paragraph("설립연도: 2020년", styles['Normal']))
    story.append(Paragraph("직원 수: 150명", styles['Normal']))
    story.append(Paragraph("본사 위치: 서울특별시 강남구", styles['Normal']))
    story.append(Paragraph("업종: 소프트웨어 개발 및 IT 컨설팅", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # 제품 정보
    story.append(Paragraph("<b>2. 주요 제품</b>", styles['Heading2']))
    
    # 제품 테이블
    product_data = [
        ['제품명', '가격', '출시일', '특징'],
        ['클라우드매니저 Pro', '월 50만원', '2021년 3월', '클라우드 자원 관리'],
        ['데이터분석 플랫폼', '월 80만원', '2022년 1월', '빅데이터 분석'],
        ['보안솔루션 Advanced', '월 120만원', '2022년 6월', '기업 보안 강화']
    ]
    
    product_table = Table(product_data, colWidths=[4*cm, 3*cm, 3*cm, 4*cm])
    product_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(product_table)
    story.append(Spacer(1, 20))
    
    # 부서 정보
    story.append(Paragraph("<b>3. 조직 구조</b>", styles['Heading2']))
    story.append(Paragraph("• 개발팀: 80명 (팀장: 김철수)", styles['Normal']))
    story.append(Paragraph("• 영업팀: 30명 (팀장: 박영희)", styles['Normal']))
    story.append(Paragraph("• 관리팀: 25명 (팀장: 이민수)", styles['Normal']))
    story.append(Paragraph("• 기획팀: 15명 (팀장: 정은지)", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # 재무 정보
    story.append(Paragraph("<b>4. 재무 현황 (2023년)</b>", styles['Heading2']))
    story.append(Paragraph("매출액: 180억원", styles['Normal']))
    story.append(Paragraph("순이익: 25억원", styles['Normal']))
    story.append(Paragraph("전년 대비 성장률: 15%", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # 연락처 정보
    story.append(Paragraph("<b>5. 연락처</b>", styles['Heading2']))
    story.append(Paragraph("전화번호: 02-1234-5678", styles['Normal']))
    story.append(Paragraph("팩스번호: 02-1234-5679", styles['Normal']))
    story.append(Paragraph("이메일: info@technosolutions.co.kr", styles['Normal']))
    story.append(Paragraph("웹사이트: www.technosolutions.co.kr", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # 추가 사실 정보
    story.append(Paragraph("<b>6. 주요 성과</b>", styles['Heading2']))
    story.append(Paragraph("• 2021년: 우수 소프트웨어 기업 선정", styles['Normal']))
    story.append(Paragraph("• 2022년: ISO 27001 인증 획득", styles['Normal']))
    story.append(Paragraph("• 2023년: 클라우드 시장 점유율 5위", styles['Normal']))
    story.append(Paragraph("• 특허 보유 건수: 12건", styles['Normal']))
    story.append(Paragraph("• 주요 고객사: 삼성전자, LG전자, 네이버", styles['Normal']))
    
    # PDF 생성
    doc.build(story)
    print(f"✅ 테스트용 PDF 생성 완료: {output_path}")
    return output_path

if __name__ == "__main__":
    create_test_pdf()
