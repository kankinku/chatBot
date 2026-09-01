from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chatbot_proxy', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='conversation',
            name='owner_key',
            field=models.CharField(
                blank=True,
                db_index=True,
                help_text='소유자 식별자',
                max_length=255,
                null=True,
            ),
        ),
        migrations.AddField(
            model_name='chatlog',
            name='owner_key',
            field=models.CharField(
                blank=True,
                db_index=True,
                help_text='소유자 식별자',
                max_length=255,
                null=True,
            ),
        ),
        migrations.AddField(
            model_name='chatmetrics',
            name='owner_key',
            field=models.CharField(
                blank=True,
                db_index=True,
                help_text='소유자 식별자',
                max_length=255,
                null=True,
            ),
        ),
    ]
