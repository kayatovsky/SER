import os
'''
Файл для конвертации датасета в нужный формат с помощью FFmpeg,
переименование (кодирование) файлов в нужный вид (файлы во всех датасетах названы в разных форматах)
'''

def convert(dir):
    names = os.listdir(dir)
    for name in names:
        fullname = os.path.join(dir, name)
        if os.path.isfile(fullname):
            dir2 = "F:\\samples\\Actor_arab\\female\\" + name
            '''
            Устанавливаем моно формат и 16 кГц (бри больших значениях librosa не справляется)
            '''
            os.system(f"ffmpeg -i {fullname} -ac 1 -ar 16000 {dir2}")


def rename(dir):
    '''
    Функция прохода по всем файлам в директории и переименования
    '''
    names = os.listdir(dir)
    i = 0
    for name in names:
        fullname = os.path.join(dir, name)
        if os.path.isfile(fullname):
            print(name[:6])
            if name[:6] == '02-05-':
                newname = '02-01-05-' + name[6:]
                name = dir + name
                newname = dir + newname
                os.rename(name, newname)
            if name[:6] == '02-03-':
                newname = '02-01-03-' + name[6:]
                name = dir + name
                newname = dir + newname
                os.rename(name, newname)
            if name[:6] == '02-04-':
                newname = '02-01-04-' + name[6:]
                name = dir + name
                newname = dir + newname
                os.rename(name, newname)
            if name[:6] == '02-01-':
                newname = '02-01-01-' + name[6:]
                name = dir + name
                newname = dir + newname
                os.rename(name, newname)
            i += 1

if __name__ == '__main__':
    rename("F:\\samples\\Actor_arab\\Actor_arabFemale\\")

