import pdfkit

path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)


input_file = r'E:\kunal\Kunal\hackathon\templates\report.html'
output_pdf = r'E:\kunal\Kunal\hackathon\pdf\report.pdf'

options = {
    'enable-local-file-access': None
}


pdfkit.from_file(input_file, output_pdf, configuration=config, options=options)

print(f'PDF saved as {output_pdf}')
