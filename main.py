import data
import traintest
import utils
from configure import config

if __name__ == '__main__':
	
	utils.set_seed(42)

	#config.setting('config/config_ecom-social_debug.json')
	config.setting('config/config_ecom-social.json')
	# config.setting('config/config_FB15k-237.json')
	# config.setting('config/config_FB15k.json')
	# config.set_log("./config/config_FB15k-237.json")
	# config.set_log("config/config_WN18.json")
	# config.set_log("config/config_WN18RR.json")
	# config.set_log("config/YAGO3-10.json")

	utils.set_logger()

	tt = traintest.TrainTest(data.KG())
	tt.train()
	tt.predict()
