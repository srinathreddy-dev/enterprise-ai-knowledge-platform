from pathlib import Path

import pymupdf


output_path = Path("data/sample/employee_handbook.pdf")
output_path.parent.mkdir(parents=True, exist_ok=True)

pages = [
    """EMPLOYEE HANDBOOK
Page 1: Paid Time Off

Full-time employees receive 15 days of paid vacation per calendar year.
Vacation requests should be submitted to the employee's manager at least
two weeks before the requested dates. Approval depends on staffing needs.

Unused vacation days do not automatically carry over into the next year.
Employees should contact Human Resources for questions about exceptions.""",

    """EMPLOYEE HANDBOOK
Page 2: Health Benefits

Eligible full-time employees may enroll in the company health insurance
plan after 30 days of employment. The company offers medical, dental,
and vision coverage options.

Employees can update benefit elections during the annual open enrollment
period or after a qualifying life event. Human Resources can explain
eligibility rules and enrollment deadlines.""",

    """EMPLOYEE HANDBOOK
Page 3: Retirement Savings

Eligible employees may participate in the company's retirement savings
plan after 90 days of employment. Employees can choose their contribution
percentage through the benefits portal.

For current employer matching details, investment options, and enrollment
instructions, employees should consult Human Resources or the official
plan documents.""",
]

with pymupdf.open() as document:
    for page_text in pages:
        page = document.new_page()
        page.insert_textbox(
            pymupdf.Rect(50, 50, 545, 750)
            page_text,
            fontsize=12,
            lineheight=1.5,
        )

    document.save(output_path)

print(f"Created sample PDF: {output_path}")
