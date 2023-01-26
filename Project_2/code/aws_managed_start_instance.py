import os
import tarfile
import time
import json
import subprocess
import argparse
import boto3

from aws_start_instance import setup_wandb, setup_team_id, build_ssh_cmd, \
    color, setup_s3_bucket, build_rsync_cmd

AWS = 'aws'   # path to `aws` CLI executable

PERMISSION_FILE_PATH = '~/.ssh/dlad-aws.pem'
AMI = 'ami-07f83f2fb8212ce3b' # Deep Learning AMI (Ubuntu 18.04) Version 41.0
INSTANCE_TYPE = 'p2.xlarge'
VOLUME_TYPE = 'gp2'
REGION = 'us-east-2'
NON_ROOT = 'ubuntu'
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))

def code_archive_filter(x):
    if 'wandb/' not in x.name and 'doc/' not in x.name and 'instance_state.txt' not in x.name and 'pycache' not in x.name and '.tar.gz' not in x.name:
        return x
    else:
        return None

def gen_code_archive(file):
    with tarfile.open(file, mode='w:gz') as tar:
        tar.add('.', filter=code_archive_filter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="config")

    args = parser.parse_args()

    setup_wandb()
    setup_s3_bucket()
    setup_team_id()

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    tag = f'{timestamp}'

    # Upload code to AWS S3
    print('Upload code to AWS S3...')
    code_archive = f'code_{tag}.tar.gz'
    gen_code_archive(code_archive)

    s3 = boto3.client('s3')
    with open('aws_configs/default_s3_bucket.txt', 'r') as fh:
        S3_BUCKET_NAME = fh.read()
    response = s3.upload_file(code_archive, S3_BUCKET_NAME, f'code/{code_archive}')
    os.remove(code_archive)

    print("Launch instance (Ctrl+C won't stop the process anymore)...")

    instance_tag = 'ResourceType=instance,Tags=[{Key=Name,Value=' + tag + '}]'
    spot_tag = 'ResourceType=spot-instances-request,Tags=[{Key=Name,Value=' + tag + '}]'
    with open('aws_configs/user_data.sh', 'r') as fh:
        user_data = fh.read().format(
            bucket=S3_BUCKET_NAME,
            code_archive=code_archive
        )

    # Refer to https://docs.aws.amazon.com/cli/latest/reference/ec2/run-instances.html
    my_cmd = [AWS, 'ec2', 'run-instances',
              '--tag-specifications', instance_tag,
              '--tag-specifications', spot_tag,
              '--instance-type', INSTANCE_TYPE,
              '--image-id', AMI,
              '--key-name', 'dlad-aws',
              '--security-groups', 'dlad-sg',
              '--iam-instance-profile', 'Name="dlad-instance-profile"',
              '--block-device-mappings', f'DeviceName="/dev/sda1",Ebs={{VolumeType="{VOLUME_TYPE}"}}',
              '--instance-market-options', f'file://{TOOLS_DIR}/aws_configs/persistent-spot-options.json',
              '--user-data', user_data,
    ]

    successful = False
    while not successful:
        try:
            response = json.loads(subprocess.check_output(my_cmd))
            successful = True
        except subprocess.CalledProcessError:
            wait_seconds = 120
            print(f'Launch unsuccessfull, retrying in {wait_seconds} seconds...')
            time.sleep(wait_seconds)

    instance_id = response['Instances'][0]['InstanceId']
    dns_response = json.loads(subprocess.check_output([
        AWS,
        'ec2',
        'describe-instances',
        '--region',
        REGION,
        '--instance-ids',
        instance_id,
    ]))
    instance_dns = dns_response['Reservations'][0]['Instances'][0]['PublicDnsName']
    ssh_command = build_ssh_cmd(instance_dns)

    print('Wait for AWS instance...')
    successful = False
    while not successful:
        try:
            subprocess.run([f"{ssh_command} echo 'SSH connection initialized'"], shell=True, check=True)
            successful = True
        except subprocess.CalledProcessError:
            print(f'Wait for instance...')

    print(f'Started instance {instance_id} with tag {tag}')
    print('Connect to instance using ssh:')
    print(color.GREEN + ssh_command + color.END)
    print('Rsync file updates:')
    print(color.GREEN + build_rsync_cmd(instance_dns) + color.END)
    print('Connect to tmux session using ssh:')
    print(color.GREEN + f"{ssh_command} -t tmux attach-session -t dlad" + color.END)
    print('When connecting with ssh to the instance, please be patient. '
          'The instance might take a few minutes to initialize the ssh server.')

    with open('aws.log', 'a') as file_name:
        file_name.write(f'{tag}\n')
        file_name.write(f'{ssh_command}\n\n')
