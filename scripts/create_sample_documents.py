from pathlib import Path

import pymupdf


OUTPUT_DIRECTORY = Path("data/sample")


DOCUMENTS = {
    "employee_benefits.pdf": {
        "title": "Employee Benefits Policy",
        "sections": [
            (
                "Health Insurance",
                """
The company provides eligible full-time employees with access to
medical, dental, and vision insurance plans.

Employees become eligible for health insurance benefits after
30 calendar days of continuous employment.

The company contributes toward the monthly medical insurance
premium. Employees are responsible for their elected share of
the premium through payroll deductions.

Employees may add eligible dependents during enrollment or
following a qualifying life event.
""",
            ),
            (
                "Dental and Vision Benefits",
                """
Eligible employees can enroll in dental and vision insurance.

Dental coverage includes preventive examinations and other
services according to the selected insurance plan.

Vision coverage may include annual eye examinations and
allowances for prescription glasses or contact lenses.

Coverage details and annual limits are defined by the applicable
insurance plan documents.
""",
            ),
            (
                "Retirement Benefits",
                """
Eligible employees may participate in the company's
retirement savings plan.

Employees can elect a contribution percentage through the
benefits enrollment system.

The company may provide matching contributions according
to the current plan rules.

Employees should consult the retirement plan documentation
for vesting schedules, contribution limits, and withdrawal rules.
""",
            ),
            (
                "Benefits Enrollment",
                """
New employees must complete benefits enrollment within
30 calendar days of their eligibility date.

Employees may review or update their benefit elections
during the annual open enrollment period.

Qualifying life events may allow changes outside the
annual enrollment period, subject to plan requirements.
""",
            ),
        ],
    },
    "it_security_policy.pdf": {
        "title": "Information Technology Security Policy",
        "sections": [
            (
                "Password Requirements",
                """
Employees must use strong, unique passwords for company
accounts and systems.

Passwords must contain at least 12 characters and should
include a combination of letters, numbers, and symbols.

Employees must never share passwords with colleagues
or store them in unapproved locations.

Compromised passwords must be reported to the IT security team
immediately.
""",
            ),
            (
                "Multi-Factor Authentication",
                """
Multi-factor authentication (MFA) is required for access
to company email, cloud applications, and remote systems.

Employees must register an approved authentication method.

Employees must never approve an authentication request
that they did not initiate.

Suspicious authentication notifications must be reported
to the IT security team.
""",
            ),
            (
                "Access Control",
                """
Access to company systems is granted according to job
responsibilities and business requirements.

Managers must request access for new employees through
the approved IT access request process.

Access permissions must be reviewed periodically.

When an employee changes roles or leaves the company,
unnecessary system access must be removed promptly.
""",
            ),
            (
                "Security Incident Reporting",
                """
Employees must immediately report suspected security
incidents to the IT security team.

Examples include phishing emails, lost company devices,
unauthorized access, and suspected data exposure.

Employees must not attempt to investigate or resolve
serious security incidents independently.

The IT security team is responsible for coordinating
incident investigation and response.
""",
            ),
            (
                "Device and Data Protection",
                """
Company devices must be protected with approved security
software and operating system updates.

Employees must lock their computers when unattended.

Confidential company information must only be stored
in approved company systems.

Employees must not transfer company data to personal
cloud storage or unauthorized external devices.
""",
            ),
        ],
    },
    "leave_attendance_policy.pdf": {
        "title": "Leave and Attendance Policy",
        "sections": [
            (
                "Paid Time Off",
                """
Eligible full-time employees receive 15 days of paid
time off (PTO) annually.

PTO may be used for vacation, personal appointments,
or other approved personal reasons.

Employees should submit planned PTO requests at least
two weeks in advance whenever practical.

PTO requests require manager approval and are subject
to operational staffing requirements.
""",
            ),
            (
                "Sick Leave",
                """
Employees who are unable to work because of illness
must notify their manager before their scheduled
workday begins whenever possible.

Employees should follow the company's absence reporting
procedure when taking sick leave.

Additional documentation may be requested where
permitted by applicable policy and law.
""",
            ),
            (
                "Attendance Expectations",
                """
Employees are expected to arrive on time and be
available during their scheduled working hours.

Repeated unexcused absences or tardiness may result
in corrective action.

Employees should notify their manager as early as
possible if they expect to arrive late.
""",
            ),
            (
                "Leave of Absence",
                """
Employees who need an extended leave of absence
should contact Human Resources.

Requests are reviewed according to company policy
and applicable legal requirements.

Employees should provide the necessary documentation
through the approved HR process.
""",
            ),
            (
                "Holiday Schedule",
                """
The company publishes an annual holiday calendar.

Employees should consult the official company calendar
for observed holidays and office closures.

Business-critical teams may have separate holiday
coverage schedules established by their managers.
""",
            ),
        ],
    },
}


def create_pdf(file_path: Path, title: str, sections: list):
    document = pymupdf.open()

    page = document.new_page()
    y_position = 72

    page.insert_text(
        (72, y_position),
        title,
        fontsize=20,
        fontname="helv",
    )
    y_position += 45

    for heading, content in sections:
        lines = content.strip().splitlines()

        required_height = 35 + len(lines) * 14

        if y_position + required_height > 740:
            page = document.new_page()
            y_position = 72

        page.insert_text(
            (72, y_position),
            heading,
            fontsize=14,
            fontname="hebo",
        )
        y_position += 25

        for line in lines:
            line = line.strip()

            if not line:
                y_position += 5
                continue

            wrapped_lines = [
                line[index:index + 85]
                for index in range(0, len(line), 85)
            ]

            for wrapped_line in wrapped_lines:
                if y_position > 750:
                    page = document.new_page()
                    y_position = 72

                page.insert_text(
                    (72, y_position),
                    wrapped_line,
                    fontsize=10,
                    fontname="helv",
                )
                y_position += 14

        y_position += 15

    document.save(file_path)
    document.close()


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for filename, content in DOCUMENTS.items():
        output_path = OUTPUT_DIRECTORY / filename

        create_pdf(
            output_path,
            content["title"],
            content["sections"],
        )

        print(f"Created: {output_path}")


if __name__ == "__main__":
    main()
