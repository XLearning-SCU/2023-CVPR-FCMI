from time import time


class TimeOperator:
    def __init__(self):
        self.time_buffer = None
        self.time_record = 0
        self.time_sum = 0
        self.time_count = 0

    def time(self, output=False, promt=''):
        if self.time_buffer is None:
            self.time_buffer = time()
        else:
            self.time_record = time() - self.time_buffer
            self.time_buffer = None
            self.time_sum += self.time_record
            self.time_count += 1
            if output:
                print('{}Time == {:7.05f}'.format(promt, self.time_record))

    def get_time_sum(self):
        return self.time_sum

    def show_time_sum(self):
        print('{:.02f}'.format(self.get_time_sum()))

    def get_fps(self):
        return self.time_count / self.time_sum

    def __get_speed(self, to_metric=None):
        speed = self.get_fps()
        metric = 'Second'
        if speed < 1 and to_metric != metric:
            speed *= 60
            metric = 'Minute'
            if speed < 1 and to_metric != metric:
                speed *= 60
                metric = 'Hour'
                if speed < 1 and to_metric != metric:
                    speed *= 24
                    metric = 'Day'
        return speed, metric

    def show_process(self, process_now, process_total, name='Epoch'):
        if self.time_sum <= 0:
            return
        speed = self.time_sum / self.time_count
        print('{:<5s} [{:3.0f}/{:3.0f}] [{:8.02f}/{:8.02f}]: {:5.02f}({:5.02f}) '.format(
            name, process_now, process_total,
            process_now * speed, process_total * speed,
            self.time_record, speed
        ))

    def show_speed(self):
        speed, metric = self.__get_speed()
        print('{:4.01f} Frames/{}'.format(speed, metric))


if __name__ == '__main__':
    print('{:03.0f}'.format(1.23456))
