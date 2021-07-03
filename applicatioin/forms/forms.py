from flask_wtf import FlaskForm
from wtforms import TextAreaField,SubmitField
from wtforms.validators import DataRequired
from flask_wtf.file import FileField, FileAllowed, FileRequired

class InputForm(FlaskForm):
	input_field_one = TextAreaField('Write An Essay')
	input_field_two = TextAreaField('Write An Essay')
	generate_report = SubmitField("Generate Report")
	check_plagiarism = SubmitField("Check Plagerism")